/*
 * avalanche.cc
 * 
 * Garfield++ simulation of a single-electron avalanche.
 *    Requires an input electric field solved by elmer and geometry from gmsh.
 *    Reads simulation parameters from runControl.
 *    Reads a run number from runNo.
 *    Saves avalanche data in a .root file as root trees:
 *        metaDataTree
 *        fieldLineDataTree
 *        gridFieldLineDataTree
 *        edgeFieldLineDataTree
 *        avalancheDataTree
 *        electronDataTree
 *        ionDataTree
 *        electronTrackDataTree
 *        ionTrackDataTree  (WIP)
 *        signalDataTree
 * 
 * Tanner Polischuk & James Harrison IV
 */

//Garfield includes
#include "Garfield/ComponentElmer.hh"
#include "Garfield/AvalancheMicroscopic.hh"
#include "Garfield/MediumMagboltz.hh"
#include "Garfield/Medium.hh"
#include "Garfield/AvalancheMC.hh"
#include "Garfield/Sensor.hh"
#include "Garfield/DriftLineRKF.hh"
#include "Garfield/ViewDrift.hh"
#include "Garfield/ViewSignal.hh"

//WIP - INDUCED SIGNALS
#include "Garfield/ViewSignal.hh"
#include "Garfield/ViewField.hh"

//ROOT includes
#include "TApplication.h"
#include "TTree.h"
#include "TFile.h"
#include "TString.h"
#include "TChain.h"
#include <TH1D.h>
#include <TCanvas.h>


//Parallelization
#include <omp.h>
#include "TROOT.h"

//C includes
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <string>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <cstdio>
#include <vector>

using namespace Garfield;

int main(int argc, char * argv[]) {
  //Random seed
  std::srand(static_cast<unsigned int>(std::time(nullptr)));

  // Enable ROOT's thread safety
  ROOT::EnableThreadSafety();

  const double MICRON = 1e-6;
  const double CM = 1e-2;
  const double MICRONTOCM = 1e-4;
  const double ELEMENTARY_CHARGE = 1.602176634e-19;
  bool DEBUG = false;
  
  //*************** SETUP ***************//
  //Timing variables
  clock_t startSim, stopSim, runTime;

  //***** Git Hash *****//
  TString gitVersion = "UNKNOWN VERSION";

  const char * getGitCommand = "git describe --tags --always --dirty 2>/dev/null";

  FILE * pipe = popen(getGitCommand, "r");
  if(!pipe){
    std::cerr << "Error: Could not open pipe to 'getGitCommand'." << std::endl;
    return -1;
  }

  char gitBuffer[128];
  std::string gitOutput = "";
  while(fgets(gitBuffer, sizeof(gitBuffer), pipe) != NULL){
    gitOutput += gitBuffer;
  }
  int gitStatus = pclose(pipe);

  if(!gitOutput.empty() && gitOutput.back() == '\n'){
    gitOutput.pop_back();
  }

  if(gitStatus != 0){
    std::cerr << "Error: 'getGitCommand' failed with status " << gitStatus << std::endl;
    return -1;
  }
  gitVersion = gitOutput;


  //***** Run numbering *****//
  //Read in run number from runNo
  int runNo;
  std::string runNoFile = "../runNo";
  std::ifstream runInFile;
  runInFile.open(runNoFile);

  if(!runInFile.is_open()){
    std::cerr << "Error reading file '" << runNoFile << "'." << std::endl;
    return -1;
  }

  runInFile >> runNo;
  runInFile.close();

  std::cout << "****************************************\n";
  std::cout << "Building simulation: " << runNo << "\n";
  std::cout << "****************************************\n";

  //Update runNo file
  std::ofstream runOutFile(runNoFile);
  if(!runOutFile.is_open()){
    std::cerr << "Error writing file '" << runNoFile << "'." << std::endl;
    return -1;
  }
  
  //Check runNo limit
  if(runNo >= 9999){
    std::cout << "Run number is large, consider resetting. (" << runNo << ")" << std::endl;
  }
  runOutFile << runNo+1;
  runOutFile.close();

  //***** Simulation Parameters *****//
  //Read in simulation parameters from runControl
  int numInputs;
  double  padLength, pitch;
  double gridStandoff, gridThickness, holeRadius;
  double cathodeHeight, thicknessSiO2, pillarRadius;
  double driftField, fieldRatio, transparencyLimit;
  int numFieldLine;
  int numAvalanche, avalancheLimit;
  double gasCompAr, gasCompCO2, gasCompCF4, gasCompIsobutane;
  double gasPenning;

  std::ifstream paramFile;
  std::string runControlFile = "../runControl";
  paramFile.open(runControlFile);

  if(!paramFile.is_open()){
    std::cerr << "Error: Could not open control file '" << runControlFile << "'." << std::endl;
    return -1;
  }

  std::cout << "****************************************\n";
  std::cout << "Setting up simulation.\n";
  std::cout << "****************************************\n";
  
  std::string curLine;
  std::map<std::string, std::string> readParam;

  //Read the file contents to a map
  int numKeys = 0;
  while(std::getline(paramFile, curLine)){
    if(curLine.find('/') == 0){
      continue;
    }

    size_t keyPos = curLine.find("=");
    if (keyPos != std::string::npos){
      std::string key = curLine.substr(0, keyPos - 1);
      std::string value = curLine.substr(keyPos + 2);
      if(value.back() == ';'){
        value.pop_back();
      }

      readParam[key] = value;
      numKeys++;
    }
  }
  paramFile.close();

  //Parse the values from the map
  numInputs = std::stoi(readParam["numInputs"]);
  if(numKeys != numInputs){
    std::cerr << "Error: Invalid simulation parameters in 'runControl'." << std::endl;
    return -1;
  }

  //Geometry parameters
  //Garfield's operational scale is cm. runControl is defined in microns
  padLength = std::stod(readParam["padLength"])*MICRONTOCM;
  pitch = std::stod(readParam["pitch"])*MICRONTOCM;

  gridStandoff = std::stod(readParam["gridStandoff"])*MICRONTOCM;
  gridThickness = std::stod(readParam["gridThickness"])*MICRONTOCM;
  holeRadius = std::stod(readParam["holeRadius"])*MICRONTOCM;

  cathodeHeight = std::stod(readParam["cathodeHeight"])*MICRONTOCM;
  thicknessSiO2 = std::stod(readParam["thicknessSiO2"])*MICRONTOCM;
  pillarRadius = std::stod(readParam["pillarRadius"])*MICRONTOCM;

  //Field parameters
  driftField = std::stod(readParam["driftField"]);
  fieldRatio = std::stod(readParam["fieldRatio"]);
  numFieldLine = std::stoi(readParam["numFieldLine"]);

  //Simulation Parameters
  numAvalanche = std::stoi(readParam["numAvalanche"]);
  avalancheLimit = std::stoi(readParam["avalancheLimit"]);

  //Gasses defined as percentages
  gasCompAr = std::stod(readParam["gasCompAr"])*100.;
  gasCompCO2 = std::stod(readParam["gasCompCO2"])*100.;
  gasCompCF4 = std::stod(readParam["gasCompCF4"])*100.;
  gasCompIsobutane = std::stod(readParam["gasCompIsobutane"])*100.;

  gasPenning = std::stod(readParam["gasPenning"]);


  // ***** Output data file ***** //  
  std::string dataFilename = "sim."+std::to_string(runNo)+".root";
  std::string dataPath = "../../Data/"+dataFilename;
  TFile *dataFile = new TFile(dataPath.c_str(), "RECREATE");

  //***** Field Line Data Tree *****//
  //Create
  TTree *fieldLineDataTree = new TTree("fieldLineDataTree", "Field Lines");

  //Data to be saved.
  int fieldLineID;
  double fieldLineX, fieldLineY, fieldLineZ;

  //Add branches
  fieldLineDataTree->Branch("Field Line ID", &fieldLineID, "fieldLineID/I");
  fieldLineDataTree->Branch("Field Line x", &fieldLineX, "fieldLineX/D");
  fieldLineDataTree->Branch("Field Line y", &fieldLineY, "fieldLineY/D");
  fieldLineDataTree->Branch("Field Line z", &fieldLineZ, "fieldLineZ/D");

  //***** Grid Field Line Data Tree *****//
  //Create
  TTree *gridFieldLineDataTree = new TTree("gridFieldLineDataTree", "Grid Field Lines");

  //Data to be saved.
  int gridFieldLineLocation;
  double gridLineX, gridLineY, gridLineZ;

  //Add branches
  gridFieldLineDataTree->Branch("Field Line ID", &fieldLineID, "fieldLineID/I");
  gridFieldLineDataTree->Branch("Grid Line Location", &gridFieldLineLocation, "gridFieldLineLocation/I");
  gridFieldLineDataTree->Branch("Field Line x", &gridLineX, "gridLineX/D");
  gridFieldLineDataTree->Branch("Field Line y", &gridLineY, "gridLineY/D");
  gridFieldLineDataTree->Branch("Field Line z", &gridLineZ, "gridLineZ/D");

  //***** Edge Field Line Data Tree *****//
  //Create
  TTree *edgeFieldLineDataTree = new TTree("edgeFieldLineDataTree", "Edge Field Lines");

  //Data to be saved.
  int edgeLineID;
  double edgeLineX, edgeLineY, edgeLineZ;

  //Add branches
  edgeFieldLineDataTree->Branch("Field Line ID", &edgeLineID, "edgeLineID/I");
  edgeFieldLineDataTree->Branch("Field Line x", &edgeLineX, "edgeLineX/D");
  edgeFieldLineDataTree->Branch("Field Line y", &edgeLineY, "edgeLineY/D");
  edgeFieldLineDataTree->Branch("Field Line z", &edgeLineZ, "edgeLineZ/D");

  
  
  //*************** SIMULATION ***************//
  std::cout << "****************************************\n";
  std::cout << "Creating simulation: " << runNo << "\n";
  std::cout << "****************************************\n";

  // Define the gas mixture
  MediumMagboltz* gasFIMS = new MediumMagboltz();

  //Set gas parameters
  gasFIMS->SetComposition(
    "ar", gasCompAr, 
    "co2", gasCompCO2,
    "cf4", gasCompCF4,
    "iC4H10", gasCompIsobutane
  );
  gasFIMS->EnablePenningTransfer(gasPenning, .0, "ar");


  //STP gas parameters:
  double gasTemperature = 293.15; //K
  double gasPressure = 760.;//torr
  int maxElectronE = 200;

  gasFIMS->SetTemperature(gasTemperature);
  gasFIMS->SetPressure(gasPressure);
  gasFIMS->SetMaxElectronEnergy(maxElectronE);
  gasFIMS->Initialise(true);

  // Load ion mobilities. 
  const std::string path = std::getenv("GARFIELD_INSTALL");
  const std::string posIonPath = path + "/share/Garfield/Data/IonMobility_Ar+_Ar.txt";
  //const std::string negIonPath = path + "/share/Garfield/Data/IonMobility_CO2+_CO2.txt";//TODO if statement here?
  const std::string negIonPath = path + "/share/Garfield/Data/IonMobility_CF4+_CF4.txt";
  gasFIMS->LoadIonMobility(posIonPath);
  gasFIMS->LoadNegativeIonMobility(negIonPath);//TODO - Is this correct for negative ion drift? 

  // Import elmer-generated field map
  std::string geometryPath = "../Geometry/";
  std::string elmerResultsPath = geometryPath+"elmerResults/";
  ComponentElmer fieldFIMS(
    elmerResultsPath+"mesh.header",
    elmerResultsPath+"mesh.elements",
    elmerResultsPath+"mesh.nodes", 
    geometryPath+"dielectrics.dat",
    elmerResultsPath+"FIMS.result", 
    "mum"
  );

  // Get region of elmer geometry
  double xmin, ymin, zmin, xmax, ymax, zmax;
  fieldFIMS.GetBoundingBox(xmin, ymin, zmin, xmax, ymax, zmax);

  //Define boundary region for simulation
  double xBoundary[2], yBoundary[2], zBoundary[2];
  zBoundary[0] = zmin;
  zBoundary[1] = zmax;
  //Extend simulation boundary to +/- pitch in x and y
  xBoundary[0] = -pitch;
  xBoundary[1] = pitch;
  yBoundary[0] = -pitch;
  yBoundary[1] = pitch;

  //Enable periodicity and set components
  fieldFIMS.EnableMirrorPeriodicityX();
  fieldFIMS.EnableMirrorPeriodicityY();
  fieldFIMS.SetGas(gasFIMS);

  // Import the weighting field for the readout electrode.
  fieldFIMS.SetWeightingField(elmerResultsPath+"FIMSWeighting.result", "centerPad");

  //Create a sensor
  Sensor* sensorFIMS = new Sensor();
  sensorFIMS->AddComponent(&fieldFIMS);
  sensorFIMS->SetArea(
    xBoundary[0], yBoundary[0], zBoundary[0], 
    xBoundary[1], yBoundary[1], zBoundary[1]
  );
  sensorFIMS->AddElectrode(&fieldFIMS, "centerPad");

  // ***** Draw field lines for visualization ***** //
  std::cout << "****************************************\n";
  std::cout << "Calculating field Lines.\n";
  std::cout << "****************************************\n";

  DriftLineRKF driftLines(sensorFIMS);
  driftLines.SetMaximumStepSize(MICRONTOCM);

  // *** Define start locations for field lines ***/
  std::vector<double> xStart;
  std::vector<double> yStart;
  std::vector<double> xEdgeStart;
  std::vector<double> yEdgeStart;
  double rangeScale = 0.99;

  // The x-direction is the long axis of the geometry. 
  const double xLineLimit = pitch/sqrt(3.);
  const double yLineLimit = pitch/2.;
  
  //Note that the total number of field lines is x2 the given number of field lines (x and y)
  // Field Lines along x:
  for(int i = 0; i < numFieldLine; i++){
    xStart.push_back(rangeScale*xLineLimit*i/(numFieldLine-1));
    yStart.push_back(0.);

    xStart.push_back(-rangeScale*xLineLimit*i/(numFieldLine-1));
    yStart.push_back(0.);
  }
  
  // Field Lines along y:
  for(int i = 0; i < numFieldLine; i++){
    xStart.push_back(0.);
    yStart.push_back(rangeScale*yLineLimit*i/(numFieldLine-1));

    xStart.push_back(0.);
    yStart.push_back(-rangeScale*yLineLimit*i/(numFieldLine-1));
  }
  
  // Lines generated along the perimeter of the unit cell
  double xWidth = rangeScale*pitch*sqrt(3.)/2.;
  double yWidth = rangeScale*pitch/2.;
  
  //Right edge of unit cell
  for(int i = 0; i < numFieldLine; i++){
    xEdgeStart.push_back(xWidth*(2./3. - (1./3.)*i/(numFieldLine-1)));
    yEdgeStart.push_back(yWidth*i/(numFieldLine-1));
  }
  //upper edge of unit cell  
  for(int i = 0; i < numFieldLine; i++){  
    xEdgeStart.push_back((1./3.)*xWidth*(1. - 1.*i/(numFieldLine-1)));
    yEdgeStart.push_back(yWidth);
  }


  // ***** Calculate field Lines ***** //
  std::vector<std::array<float, 3> > fieldLines;
  int totalFieldLines = xStart.size();
  int totalEdgeFieldLines = xEdgeStart.size();
  std::cout << "Computing " << totalFieldLines << " field lines along the x and y axes." << std::endl;

  // Note that true number is 3x totalFieldLines - Cathode, above, and below grid
  int prevDriftLine = 0;
  for(int inFieldLine = 0; inFieldLine < totalFieldLines; inFieldLine++){
    
    fieldLineID = inFieldLine;

    // Calculate from top of volume
    fieldLines.clear();
    driftLines.FieldLine(xStart[inFieldLine], yStart[inFieldLine], zmax*.95, fieldLines);

    // Get coordinates of every point along field line and fill the tree
    for(int inLine = 0; inLine < fieldLines.size(); inLine++){
      fieldLineX = fieldLines[inLine][0];
      fieldLineY = fieldLines[inLine][1];
      fieldLineZ = fieldLines[inLine][2];

      fieldLineDataTree->Fill();
    }

    //Calculate lines from grid - only do those outside of hole
    double lineRadius2 = std::pow(xStart[inFieldLine], 2.) + std::pow(yStart[inFieldLine], 2.);
    double holeRadius2 = std::pow(holeRadius, 2.);
    double gridLineSeparation = 2.0;

    //Do above grid
    gridFieldLineLocation = 1;
    fieldLines.clear();

    if(lineRadius2 >= holeRadius2){
      driftLines.FieldLine(xStart[inFieldLine], yStart[inFieldLine], gridLineSeparation*gridThickness/2., fieldLines);

      //Get coordinates of every point along field line and fill the tree
      for(int inLine = 0; inLine < fieldLines.size(); inLine++){
        gridLineX = fieldLines[inLine][0];
        gridLineY = fieldLines[inLine][1];
        gridLineZ = fieldLines[inLine][2];

        gridFieldLineDataTree->Fill();
      }
    }
    
    //Do below grid
    gridFieldLineLocation = -1;
    fieldLines.clear();

    if(lineRadius2 >= holeRadius2){
      driftLines.FieldLine(xStart[inFieldLine], yStart[inFieldLine], -gridLineSeparation*gridThickness/2., fieldLines);

      //Get coordinates of every point along field line and fill the tree
      for(int inLine = 0; inLine < fieldLines.size(); inLine++){
        gridLineX = fieldLines[inLine][0];
        gridLineY = fieldLines[inLine][1];
        gridLineZ = fieldLines[inLine][2];

        gridFieldLineDataTree->Fill();
      }
    }
    
    //Print a progress update every 10%
    int driftLineProgress = (100*(inFieldLine+1))/totalFieldLines;
    if((driftLineProgress % 10 == 0) && (driftLineProgress != prevDriftLine)){
      std::cout << "Driftline Progress: " << driftLineProgress << " %" << std::endl;
      prevDriftLine = driftLineProgress;
    }
  }// End field lines loop
  
  prevDriftLine = 0;
  for(int inEdgeFieldLine = 0; inEdgeFieldLine < totalEdgeFieldLines; inEdgeFieldLine++){
    
    edgeLineID = inEdgeFieldLine;

    // Calculate along the perimeter
    fieldLines.clear();
    driftLines.FieldLine(xEdgeStart[inEdgeFieldLine], yEdgeStart[inEdgeFieldLine], zmax*.95, fieldLines);

    // Get coordinates of every point along field line and fill the tree
    for(int inLine = 0; inLine < fieldLines.size(); inLine++){
      edgeLineX = fieldLines[inLine][0];
      edgeLineY = fieldLines[inLine][1];
      edgeLineZ = fieldLines[inLine][2];
      
      edgeFieldLineDataTree->Fill();
    }
    
    //Print a progress update every 10%
    int driftLineProgress = (100*(inEdgeFieldLine+1))/totalEdgeFieldLines;
    if((driftLineProgress % 10 == 0) && (driftLineProgress != prevDriftLine)){
      std::cout << "Edge Driftline Progress: " << driftLineProgress << " %" << std::endl;
      prevDriftLine = driftLineProgress;
    }
    
  }//End edge field line loop
  
  std::cout << "Done " << totalFieldLines << " field lines." << std::endl;

  // ***** Deal with fieldline data trees ***** //
  fieldLineDataTree->Write();
  delete fieldLineDataTree;
  gridFieldLineDataTree->Write();
  delete gridFieldLineDataTree;
  edgeFieldLineDataTree->Write();
  delete edgeFieldLineDataTree;

  dataFile->Close();
  delete dataFile;
  
  
  // ***** Prepare Avalanche Electron ***** //
  //Set the Initial electron parameters
  double x0 = 0., y0 = 0., z0 = holeRadius;
  double t0 = 0.;//ns
  double e0 = 0.1;//eV (Garfield is weird when this is 0.)
  double dx0 = 0., dy0 = 0., dz0 = 0.;//No velocity

  double timeFinal = 5.;//ns
  double timeStep = 0.01;//ns
  int nSignalBins = timeFinal/timeStep;
  
  //Start timing the sim
  startSim = clock();

  if(numAvalanche == 0){
    std::cerr << "No avalanches - Defaulting to 100." << std::endl;
    numAvalanche = 100;
  }
  std::cout << "****************************************\n";
  std::cout << "Starting simulation: " << runNo << "\n";
  std::cout << "****************************************\n";

  std::cout << "Starting " << numAvalanche << " avalanches." << std::endl;

  //*** Set up parallel avalanche loops ***//
  std::vector<std::string> parallelFileNames;
  #pragma omp parallel
  {
    //thread-local pointers
    ComponentElmer* parallelFieldFIMS = nullptr;
    Sensor* parallelSensorFIMS = nullptr;
    AvalancheMicroscopic* avalancheE = nullptr;
    AvalancheMC* driftIon = nullptr;
    ViewDrift* viewElectronDrift = nullptr;
    ViewDrift* viewIonDrift = nullptr;
    TFile* parallelDataFile = nullptr;
    std::string parallelFilename;

    // Create thread-local objects
    #pragma omp critical
    {//Critical for file I/O

      //Create objects for this thread
      parallelFieldFIMS = new ComponentElmer(
        elmerResultsPath+"mesh.header",
        elmerResultsPath+"mesh.elements",
        elmerResultsPath+"mesh.nodes", 
        geometryPath+"dielectrics.dat",
        elmerResultsPath+"FIMS.result", 
        "mum"
      );
      parallelSensorFIMS = new Sensor();
      avalancheE = new AvalancheMicroscopic;
      driftIon = new AvalancheMC;
      viewElectronDrift = new ViewDrift();
      viewIonDrift = new ViewDrift();

      //Link objects
      parallelFieldFIMS->SetGas(gasFIMS);
      parallelFieldFIMS->SetWeightingField(elmerResultsPath+"FIMSWeighting.result", "centerPad");
      parallelFieldFIMS->EnableMirrorPeriodicityX();
      parallelFieldFIMS->EnableMirrorPeriodicityY();
      
      parallelSensorFIMS->AddComponent(parallelFieldFIMS);
      parallelSensorFIMS->SetArea(
        xBoundary[0], yBoundary[0], zBoundary[0], 
        xBoundary[1], yBoundary[1], zBoundary[1]
      );      

      parallelSensorFIMS->AddElectrode(parallelFieldFIMS, "centerPad");
      parallelSensorFIMS->SetTimeWindow(0., timeStep, nSignalBins);

      avalancheE->SetSensor(parallelSensorFIMS);
      avalancheE->EnableAvalancheSizeLimit(avalancheLimit);

      driftIon->SetSensor(parallelSensorFIMS);
      driftIon->SetDistanceSteps(MICRONTOCM/10.);
      driftIon->EnableDriftLines(true);
      
      viewElectronDrift->SetArea(
        xBoundary[0], yBoundary[0], zBoundary[0], 
        xBoundary[1], yBoundary[1], zBoundary[1]
      );
      viewIonDrift->SetArea(
        xBoundary[0], yBoundary[0], zBoundary[0], 
        xBoundary[1], yBoundary[1], zBoundary[1]
      );
      
      
      avalancheE->EnablePlotting(viewElectronDrift, 250);
      driftIon->EnablePlotting(viewIonDrift);
      
      //Filename
      int threadID = omp_get_thread_num();
      std::string parallelDataPath = "parallelData/";
      std::string parallelRunNo = "parallelSim.";
      std::string parallelThreadNo = std::to_string(threadID);

      parallelFilename = parallelDataPath + parallelRunNo + parallelThreadNo + ".root";
      parallelFileNames.push_back(parallelFilename);

      parallelDataFile = new TFile(parallelFilename.c_str(), "RECREATE");

    }//end critical

    //Variables for trees
    int avalancheID;
    bool hitLimit;
    int totalElectrons, attachedElectrons, totalIons;
    int electronID;
    double xi, yi, zi, ti, Ei;
    double xf, yf, zf, tf, Ef;
    int stat;
    int ionCharge;
    double xiIon, yiIon, ziIon, tiIon;
    double xfIon, yfIon, zfIon, tfIon;
    int statIon;
    float electronDriftx, electronDrifty, electronDriftz;
    float ionDriftx, ionDrifty, ionDriftz, ionDriftt;
    double signalTime, signalStrength;

    TTree* parallelAvalancheDataTree = new TTree("avalancheDataTree", "Avalanche Results");
    parallelAvalancheDataTree->Branch("Avalanche ID", &avalancheID, "avalancheID/I");
    parallelAvalancheDataTree->Branch("Reached Limit", &hitLimit, "hitLimit/B");
    parallelAvalancheDataTree->Branch("Total Electrons", &totalElectrons, "totalElectrons/I");
    parallelAvalancheDataTree->Branch("Attached Electrons", &attachedElectrons, "attachedElectrons/I");
    parallelAvalancheDataTree->Branch("Total Ions", &totalIons, "totalIons/I");

    TTree* parallelElectronDataTree = new TTree("electronDataTree", "Avalanche Electron Parameters");
    parallelElectronDataTree->Branch("Avalanche ID", &avalancheID, "avalancheID/I");
    parallelElectronDataTree->Branch("Electron ID", &electronID, "electronID/I");
    parallelElectronDataTree->Branch("Initial x", &xi, "xi/D");
    parallelElectronDataTree->Branch("Initial y", &yi, "yi/D");
    parallelElectronDataTree->Branch("Initial z", &zi, "zi/D");
    parallelElectronDataTree->Branch("Initial Time", &ti, "ti/D");
    parallelElectronDataTree->Branch("Initial Energy", &Ei, "Ei/D");
    parallelElectronDataTree->Branch("Final x", &xf, "xf/D");
    parallelElectronDataTree->Branch("Final y", &yf, "yf/D");
    parallelElectronDataTree->Branch("Final z", &zf, "zf/D");
    parallelElectronDataTree->Branch("Final Time", &tf, "tf/D");
    parallelElectronDataTree->Branch("Final Energy", &Ef, "Ef/D");
    parallelElectronDataTree->Branch("Exit Status", &stat, "stat/I");

    TTree *parallelIonDataTree = new TTree("ionDataTree", "Avalanche Ion Parameters");
    parallelIonDataTree->Branch("Avalanche ID", &avalancheID, "avalancheID/I");
    parallelIonDataTree->Branch("Electron ID", &electronID, "electronID/I");
    parallelIonDataTree->Branch("Ion Charge", &ionCharge, "ionCharge/I");
    parallelIonDataTree->Branch("Initial x", &xiIon, "xiIon/D");
    parallelIonDataTree->Branch("Initial y", &yiIon, "yiIon/D");
    parallelIonDataTree->Branch("Initial z", &ziIon, "ziIon/D");
    parallelIonDataTree->Branch("Initial Time", &tiIon, "tiIon/D");
    parallelIonDataTree->Branch("Final x", &xfIon, "xfIon/D");
    parallelIonDataTree->Branch("Final y", &yfIon, "yfIon/D");
    parallelIonDataTree->Branch("Final z", &zfIon, "zfIon/D");
    parallelIonDataTree->Branch("Final Time", &tfIon, "tfIon/D");
    parallelIonDataTree->Branch("Exit Status", &statIon, "statIon/I");

    TTree* parallelElectronTrackDataTree = new TTree("electronTrackDataTree", "Electron Tracks");
    parallelElectronTrackDataTree->Branch("Avalanche ID", &avalancheID, "avalancheID/I");
    parallelElectronTrackDataTree->Branch("Electron ID", &electronID, "electronID/I");
    parallelElectronTrackDataTree->Branch("Drift x", &electronDriftx, "electronDriftx/F");
    parallelElectronTrackDataTree->Branch("Drift y", &electronDrifty, "electronDrifty/F");
    parallelElectronTrackDataTree->Branch("Drift z", &electronDriftz, "electronDriftz/F");
  
    TTree* parallelIonTrackDataTree = new TTree("ionTrackDataTree", "Ion Tracks");
    parallelIonTrackDataTree->Branch("Avalanche ID", &avalancheID, "avalancheID/I");
    parallelIonTrackDataTree->Branch("Electron ID", &electronID, "electronID/I");
    parallelIonTrackDataTree->Branch("Drift x", &ionDriftx, "ionDriftx/F");
    parallelIonTrackDataTree->Branch("Drift y", &ionDrifty, "ionDrifty/F");
    parallelIonTrackDataTree->Branch("Drift z", &ionDriftz, "ionDriftz/F");

    TTree* parallelSignalDataTree = new TTree("signalDataTree", "Induced Signal");
    parallelSignalDataTree->Branch("Avalanche ID", &avalancheID, "avalancheID/I");
    parallelSignalDataTree->Branch("Signal Time", &signalTime, "signalTime/D");
    parallelSignalDataTree->Branch("Signal Strength", &signalStrength, "signalStrength/D");
  

    //***** Parallel Avalanche Loop *****//
    #pragma omp for schedule(dynamic)
    for(int inAvalanche = 0; inAvalanche < numAvalanche; inAvalanche++){
      if(DEBUG){
        continue;
      }
      
      avalancheID = inAvalanche;

      //Reset avalanche data
      totalElectrons = 0;
      attachedElectrons = 0;
      totalIons = 0;
      
      //Begin single-electron avalanche
      avalancheE->AvalancheElectron(x0, y0, z0, 0., e0, dx0, dy0, dz0);

      //Electron count - use endpoints to include attached electrons
      int avalancheElectrons = avalancheE->GetNumberOfElectronEndpoints();

      //Check if avalanche limit was reached
      if(avalancheElectrons >= avalancheLimit){
        hitLimit = true;
      }
      else{
        hitLimit = false;
      }

      //Loop through all electrons in avalanche
      for(int inElectron = 0; inElectron < avalancheElectrons; inElectron++){
        electronID = inElectron;

        //Extract individual electron data
        avalancheE->GetElectronEndpoint(inElectron, xi, yi, zi, ti, Ei, xf, yf, zf, tf, Ef, stat);
          
        totalElectrons++;

        ionCharge = 1;
        driftIon->DriftIon(xi, yi, zi, ti);
        
        driftIon->GetIonEndpoint(0, xiIon, yiIon, ziIon, tiIon, xfIon, yfIon, zfIon, tfIon, statIon);
        
        //Fill tree with data from this positive ion
        parallelIonDataTree->Fill();
        
        //get ion drift lines
        bool isIon;
        std::vector<std::array<float, 3> > ionDriftLines;
        
        viewIonDrift->GetDriftLine(0, ionDriftLines, isIon);
        
        for(int ionPoint = 0; ionPoint < ionDriftLines.size(); ionPoint++){
          ionDriftx = ionDriftLines[ionPoint][0];
          ionDrifty = ionDriftLines[ionPoint][1];
          ionDriftz = ionDriftLines[ionPoint][2];
          //Fill tree with data for this point
          parallelIonTrackDataTree->Fill();
        }
        
        totalIons++;
        viewIonDrift->Clear();

        //Check for electron attachment
        if(stat == -7){
          attachedElectrons++;

          //Drift negative ion from end of electron tracks that attach
          ionCharge = -1;
          driftIon->DriftNegativeIon(xf, yf, zf, tf);
          driftIon->GetNegativeIonEndpoint(0, xiIon, yiIon, ziIon, tiIon, xfIon, yfIon, zfIon, tfIon, statIon);

          totalIons++;
        }

        // Get electron drift line data
        int numElectronDrift = viewElectronDrift->GetNumberOfDriftLines();
        bool isElectron;
        std::vector<std::array<float, 3> > electronDriftLines;
        viewElectronDrift->GetDriftLine(inElectron, electronDriftLines, isElectron);
        for(int inPoint = 0; inPoint < electronDriftLines.size(); inPoint++){
          electronDriftx = electronDriftLines[inPoint][0];
          electronDrifty = electronDriftLines[inPoint][1];
          electronDriftz = electronDriftLines[inPoint][2];

          //Fill tree with data from this point
          parallelElectronTrackDataTree->Fill();
        }

        //*** TODO ***/
        //Can insert any per-electron analysis/data here.
        // --Velocity?

        //Fill tree with data from this electron
        parallelElectronDataTree->Fill();

      }//end electrons in avalanche loop

      //Get signal for each timestep
      for(int inSignal = 0; inSignal < nSignalBins; inSignal++){
        signalTime = inSignal*timeStep;
        signalStrength = parallelSensorFIMS->GetSignal("centerPad", inSignal);
        
        //Fill tree
        parallelSignalDataTree->Fill();
      }
      parallelSensorFIMS->ClearSignal();

      //*** TODO ***/
      //Can insert any other per-avalanche analysis/data here.
      // -- Histograms of energy loss/collison, time between collisions,

      //Fill tree with data from this avalanche
      parallelAvalancheDataTree->Fill();
      //clean up memory
      viewElectronDrift->Clear();

    }//end avalanche loop


    // Write and close the file.
    parallelDataFile->Write();
    parallelDataFile->Close();

    delete parallelSensorFIMS;
    delete avalancheE;
    delete driftIon;
    delete viewElectronDrift;
    delete viewIonDrift;
    delete parallelDataFile;

  }//End parallization


  std::cout << "****************************************\n";
  std::cout << "Done avalanches for run: " << runNo << "\n";
  std::cout << "Getting diffusion coefficients...\n";
  std::cout << "****************************************\n";


  //Calculate diffusion coefficients
  double vx, vy, wv, wr;
  double alpha, eta, riontof, ratttof, lor;
  double vxerr, vyerr, vzerr, wverr, wrerr, dlerr, dterr;
  double alphaerr, etaerr, riontoferr, ratttoferr, lorerr, alphatof;
  std::array<double, 6> difftens;

  // Drift field
  double driftDiffusionL, driftDiffusionT, driftVelocity;

  gasFIMS->RunMagboltz(
    driftField, 0., 0., 1, true,
    vx, vy, driftVelocity, wv, wr, 
    driftDiffusionL, driftDiffusionT,
    alpha, eta, riontof, ratttof, lor, 
    vxerr, vyerr, vzerr, wverr, wrerr, dlerr, dterr,
    alphaerr, etaerr, riontoferr, ratttoferr, lorerr, alphatof,
    difftens
  );

  //Amplification field
  double ampDiffusionL, ampDiffusionT, ampVelocity;
  double ampField = driftField*fieldRatio;

  gasFIMS->RunMagboltz(
    ampField, 0., 0., 1, true,
    vx, vy, ampVelocity, wv, wr, 
    ampDiffusionL, ampDiffusionT,
    alpha, eta, riontof, ratttof, lor, 
    vxerr, vyerr, vzerr, wverr, wrerr, dlerr, dterr,
    alphaerr, etaerr, riontoferr, ratttoferr, lorerr, alphatof,
    difftens
  );

  delete gasFIMS;

  //Final timing
  stopSim = clock();
  runTime = (stopSim - startSim)/CLOCKS_PER_SEC;
  std::cout << "****************************************\n";
  std::cout << "Done processing avalanches...(" << runTime << " s)\n";
  std::cout << "****************************************\n";

  //***** Deal with Root trees and files *****//

  // ***** Metadata tree ***** //
  //Fill the meta data tree
  TTree *metaDataTree = new TTree("metaDataTree", "Simulation Parameters");

  metaDataTree->Branch("Git Version", &gitVersion);
  metaDataTree->Branch("runNo", &runNo, "runNo/I");

  metaDataTree->Branch("Pad Length", &padLength, "padLength/D");
  metaDataTree->Branch("Pitch", &pitch, "pitch/D");
  metaDataTree->Branch("Grid Standoff", &gridStandoff, "gridStandoff/D");
  metaDataTree->Branch("Grid Thickness", &gridThickness, "gridThickness/D");
  metaDataTree->Branch("Hole Radius", &holeRadius, "holeRadius/D");
  metaDataTree->Branch("Cathode Height", &cathodeHeight, "cathodeHeight/D");
  metaDataTree->Branch("Thickness SiO2", &thicknessSiO2, "thicknessSiO2/D");
  metaDataTree->Branch("Pillar Radius", &pillarRadius, "pillarRadius/D");

  metaDataTree->Branch("Electric Field Ratio", &fieldRatio, "fieldRatio/D");
  metaDataTree->Branch("Drift Field", &driftField, "driftField/D");
  metaDataTree->Branch("Amplification Field", &ampField, "ampField/D");

  metaDataTree->Branch("Number of Field Lines", &numFieldLine, "numFieldLine/I");
  metaDataTree->Branch("Transparency Limit", &transparencyLimit, "transparencyLimit/D");
  metaDataTree->Branch("Number of Avalanches", &numAvalanche, "numAvalanche/I");
  metaDataTree->Branch("Avalanche Limit", &avalancheLimit, "avalancheLimit/I");
  
  metaDataTree->Branch("Gas Comp: Ar", &gasCompAr, "gasCompAr/D");
  metaDataTree->Branch("Gas Comp: CO2", &gasCompCO2, "gasCompCO2/D");
  metaDataTree->Branch("Gas Comp: CF4", &gasCompCF4, "gasCompCF4/D");
  metaDataTree->Branch("Gas Comp: Isobutane", &gasCompIsobutane, "gasCompIsobutane/D");
  metaDataTree->Branch("Gas Penning", &gasPenning, "gasPenning/D");

  metaDataTree->Branch("Drift Velocity (Drift) ", &driftVelocity, "driftVelocity/D");
  metaDataTree->Branch("Diffusion L (Drift)", &driftDiffusionL, "driftDiffusionL/D");
  metaDataTree->Branch("Diffusion T (Drift)", &driftDiffusionT, "driftDiffusionT/D");
  
  metaDataTree->Branch("Drift Velocity (Amplify) ", &ampVelocity, "ampVelocity/D");
  metaDataTree->Branch("Diffusion L (Amplify)", &ampDiffusionL, "ampDiffusionL/D");
  metaDataTree->Branch("Diffusion T (Amplify)", &ampDiffusionT, "ampDiffusionT/D");

  metaDataTree->Fill();
  
  // ***** Deal with data ***** //

  //Reopen file and write metadata
  dataFile = new TFile(dataPath.c_str(), "UPDATE");

  metaDataTree->Write();
  delete metaDataTree;

  // Deal with parallel trees
  std::vector<std::string> treeNames = {
    "avalancheDataTree",
    "electronDataTree",
    "ionDataTree",
    "electronTrackDataTree",
    "ionTrackDataTree",
    "signalDataTree"
  };

  std::cout << "Merging parallel trees...\n";
  for(const auto& inTree : treeNames){

    //Chain the trees together
    TChain treeChain(inTree.c_str());
    for(const auto& filename : parallelFileNames){
      treeChain.Add(filename.c_str());
    }

    TTree* newTree = treeChain.CloneTree(-1, "fast");
    if(!newTree){
      std::cout << "Error combining parallel tree " << inTree.c_str() << std::endl;
    }
    newTree->Write();
    delete newTree;

  }

  dataFile->Close();
  delete dataFile;

  // Clean up parallel thread files
  for(const auto& filename : parallelFileNames) {
    std::remove(filename.c_str());
  }

  std::cout << "****************************************\n";
  std::cout << "Done simulation: " << runNo << "\n";
  std::cout << "****************************************\n";
  std::cout << std::endl;

  return 0;

}
