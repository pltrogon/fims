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
 *        avalancheDataTree
 *        electronDataTree
 *        ionDataTree
 *        electronTrackDataTree
 * 
 * Tanner Polischuk & James Harrison IV
 */

//Garfield includes
#include "Garfield/ComponentElmer.hh"
#include "Garfield/AvalancheMicroscopic.hh"
#include "Garfield/MediumMagboltz.hh"
#include "Garfield/AvalancheMC.hh"
#include "Garfield/Sensor.hh"
#include "Garfield/DriftLineRKF.hh"
#include "Garfield/ViewDrift.hh"

//ROOT includes
#include "TApplication.h"
#include "TTree.h"
#include "TFile.h"
#include "TString.h"

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

using namespace Garfield;

int main(int argc, char * argv[]) {

  std::srand(static_cast<unsigned int>(std::time(nullptr)));

  const double MICRON = 1e-6;
  const double CM = 1e-2;
  const double MICRONTOCM = 1e-4;
  bool DEBUG = false;
  
  //*************** SETUP ***************//
  //Timing variables
  clock_t startSim, stopSim, lapAvalanche, runTime;

  TApplication app("app", &argc, argv);

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

  //***** Data File *****//
  std::string dataFilename = "sim."+std::to_string(runNo)+".root";
  std::string dataPath = "../../Data/"+dataFilename;
  TFile *dataFile = new TFile(dataPath.c_str(), "NEW");

  if(!dataFile->IsOpen()){
    std::cerr << "Error creating or opening file '" << dataFilename << "'." << std::endl;
    return -1;
  }  
  
  std::cout << "File '" << dataFilename << "' created and opened successfully.\n";


  //***** Simulation Parameters *****//
  //Read in simulation parameters from runControl

  double  padLength, pitch;
  double gridStandoff, gridThickness, holeRadius;
  double cathodeHeight, thicknessSiO2;
  double fieldRatio, transparencyLimit;
  int numFieldLine;
  int numAvalanche, avalancheLimit;
  double gasCompAr, gasCompCO2;

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
  if(numKeys != 14){//Number of user-defined simulation parameters in runControl to search for.
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

  //Field parameters
  fieldRatio = std::stod(readParam["fieldRatio"]);
  transparencyLimit = std::stod(readParam["transparencyLimit"]);

  numFieldLine = std::stoi(readParam["numFieldLine"]);

  //Simulation Parameters
  numAvalanche = std::stoi(readParam["numAvalanche"]);
  avalancheLimit = std::stoi(readParam["avalancheLimit"]);

  gasCompAr = std::stod(readParam["gasCompAr"]);
  gasCompCO2 = std::stod(readParam["gasCompCO2"]);


  // ***** Metadata tree ***** //
  //Create
  TTree *metaDataTree = new TTree("metaDataTree", "Simulation Parameters");

  //Data to be saved

  //Add branches
  //General
  metaDataTree->Branch("Git Version", &gitVersion);
  metaDataTree->Branch("runNo", &runNo, "runNo/I");

  //Geometry Parameters
  metaDataTree->Branch("Pad Length", &padLength, "padLength/D");
  metaDataTree->Branch("Pitch", &pitch, "pitch/D");
  metaDataTree->Branch("Grid Standoff", &gridStandoff, "gridStandoff/D");
  metaDataTree->Branch("Grid Thickness", &gridThickness, "gridThickness/D");
  metaDataTree->Branch("Hole Radius", &holeRadius, "holeRadius/D");
  metaDataTree->Branch("Cathode Height", &cathodeHeight, "cathodeHeight/D");
  metaDataTree->Branch("Thickness SiO2", &thicknessSiO2, "thicknessSiO2/D");

  //Field Parameters
  metaDataTree->Branch("Electric Field Ratio", &fieldRatio, "fieldRatio/D");
  metaDataTree->Branch("Number of Field Lines", &numFieldLine, "numFieldLine/I");
  metaDataTree->Branch("Transparency Limit", &transparencyLimit, "transparencyLimit/D");

  //Simulation parameters
  metaDataTree->Branch("Number of Avalanches", &numAvalanche, "numAvalanche/I");
  metaDataTree->Branch("Avalanche Limit", &avalancheLimit, "avalancheLimit/I");
  metaDataTree->Branch("Gas Comp: Ar", &gasCompAr, "gasCompAr/D");
  metaDataTree->Branch("Gas Comp: CO2", &gasCompCO2, "gasCompCO2/D");

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

  //***** Avalanche Data Tree *****//
  //Create
  TTree *avalancheDataTree = new TTree("avalancheDataTree", "Avalanche Results");

  //Data to be saved for each avalanche
  int avalancheID;
  bool hitLimit;//T/F if avalanche limit was hit
  int totalElectrons, attachedElectrons, totalIons;

  //Add Branches
  avalancheDataTree->Branch("Avalanche ID", &avalancheID, "avalancheID/I");
  avalancheDataTree->Branch("Reached Limit", &hitLimit, "hitLimit/B");
  avalancheDataTree->Branch("Total Electrons", &totalElectrons, "totalElectrons/I");
  avalancheDataTree->Branch("Attached Electrons", &attachedElectrons, "attachedElectrons/I");
  avalancheDataTree->Branch("Total Ions", &totalIons, "totalIons/I");

  //***** Electron Data Tree *****//

  //Create
  TTree *electronDataTree = new TTree("electronDataTree", "Avalanche Electron Parameters");

  // Data to be saved for each electron
  int electronID;
  double xi, yi, zi, ti, Ei; //Initial parameters
  double xf, yf, zf, tf, Ef; //Final parameters
  int stat; // Electron status

  //Add Branches
  electronDataTree->Branch("Avalanche ID", &avalancheID, "avalancheID/I");
  electronDataTree->Branch("Electron ID", &electronID, "electronID/I");

  electronDataTree->Branch("Initial x", &xi, "xi/D");
  electronDataTree->Branch("Initial y", &yi, "yi/D");
  electronDataTree->Branch("Initial z", &zi, "zi/D");
  electronDataTree->Branch("Initial Time", &ti, "ti/D");
  electronDataTree->Branch("Initial Energy", &Ei, "Ei/D");

  electronDataTree->Branch("Final x", &xf, "xf/D");
  electronDataTree->Branch("Final y", &yf, "yf/D");
  electronDataTree->Branch("Final z", &zf, "zf/D");
  electronDataTree->Branch("Final Time", &tf, "tf/D");
  electronDataTree->Branch("Final Energy", &Ef, "Ef/D");

  electronDataTree->Branch("Exit Status", &stat, "stat/I");

  //***** Ion Data Tree *****//

  //Create
  TTree *ionDataTree = new TTree("ionDataTree", "Avalanche Ion Parameters");

  // Data to be saved for each Ion
  int ionCharge;
  double xiIon, yiIon, ziIon, tiIon; //Initial parameters
  double xfIon, yfIon, zfIon, tfIon; //Final parameters
  int statIon;

  //Add Branches
  ionDataTree->Branch("Avalanche ID", &avalancheID, "avalancheID/I");
  ionDataTree->Branch("Electron ID", &electronID, "electronID/I");
  ionDataTree->Branch("Ion Charge", &ionCharge, "ionCharge/I");

  ionDataTree->Branch("Initial x", &xiIon, "xiIon/D");
  ionDataTree->Branch("Initial y", &yiIon, "yiIon/D");
  ionDataTree->Branch("Initial z", &ziIon, "ziIon/D");
  ionDataTree->Branch("Initial Time", &tiIon, "tiIon/D");

  ionDataTree->Branch("Final x", &xfIon, "xfIon/D");
  ionDataTree->Branch("Final y", &yfIon, "yfIon/D");
  ionDataTree->Branch("Final z", &zfIon, "zfIon/D");
  ionDataTree->Branch("Final Time", &tfIon, "tfIon/D");

  ionDataTree->Branch("Exit Status", &statIon, "statIon/I");


  //***** Electron Track Data Tree *****/
  //Create
  TTree *electronTrackDataTree = new TTree("electronTrackDataTree", "Electron Tracks");

  // Data to be saved for each Electron track
  float electronDriftx, electronDrifty, electronDriftz;

  //Add Branches
  electronTrackDataTree->Branch("Avalanche ID", &avalancheID, "avalancheID/I");
  electronTrackDataTree->Branch("Electron ID", &electronID, "electronID/I");

  electronTrackDataTree->Branch("Drift x", &electronDriftx, "electronDriftx/F");
  electronTrackDataTree->Branch("Drift y", &electronDrifty, "electronDrifty/F");
  electronTrackDataTree->Branch("Drift z", &electronDriftz, "electronDriftz/F");


  //*************** SIMULATION ***************//
  std::cout << "****************************************\n";
  std::cout << "Creating simulation: " << runNo << "\n";
  std::cout << "****************************************\n";

  // Define the gas mixture
  MediumMagboltz* gasFIMS = new MediumMagboltz();

  //Set parameters
  gasFIMS->SetComposition("ar", gasCompAr, "co2", gasCompCO2);
  gasFIMS->SetTemperature(293.15); // Room temperature
  gasFIMS->SetPressure(760.);     // Atmospheric pressure
  gasFIMS->SetMaxElectronEnergy(200);
  gasFIMS->Initialise(true);
  // Load the penning transfer and ion mobilities.
  gasFIMS->EnablePenningTransfer(0.51, .0, "ar");

  const std::string path = std::getenv("GARFIELD_INSTALL");
  gasFIMS->LoadIonMobility(path + "/share/Garfield/Data/IonMobility_Ar+_Ar.txt");
  gasFIMS->LoadNegativeIonMobility(path + "/share/Garfield/Data/IonMobility_CO2+_CO2.txt");//TODO - Is this correct for negative ion

  // Import elmer-generated field map
  std::string geometryPath = "../Geometry/";
  std::string elmerResultsPath = geometryPath+"elmerResults/";
  ComponentElmer fieldFIMS(elmerResultsPath+"mesh.header",
                           elmerResultsPath+"mesh.elements",
                           elmerResultsPath+"mesh.nodes", 
                           geometryPath+"dielectrics.dat",
                           elmerResultsPath+"FIMS.result", 
                           "mum");

  // Get region of elmer geometry
  double xmin, ymin, zmin, xmax, ymax, zmax;
  fieldFIMS.GetBoundingBox(xmin, ymin, zmin, xmax, ymax, zmax);

  //Define boundary region for simulation
  double xBoundary[2], yBoundary[2], zBoundary[2];

  //Simple criteria for if 1/4 geometry or full
  //   x=y=0 should be the min if 1/4
  if(xmin == 0 && ymin == 0){
    xBoundary[0] = -xmax;
    xBoundary[1] = xmax;
    yBoundary[0] = -ymax;
    yBoundary[1] = ymax;
  }
  else{
    xBoundary[0] = xmin;
    xBoundary[1] = xmax;
    yBoundary[0] = ymin;
    yBoundary[1] = ymax;
  }
  zBoundary[0] = zmin;
  zBoundary[1] = zmax;

  //Enable periodicity and set components
  fieldFIMS.EnableMirrorPeriodicityX();
  fieldFIMS.EnableMirrorPeriodicityY();
  fieldFIMS.SetGas(gasFIMS);

  // Import the weighting field for the readout electrode.
  fieldFIMS.SetWeightingField(elmerResultsPath+"FIMSWeighting.result", "wtlel");

  //Create a sensor
  Sensor* sensorFIMS = new Sensor();
  sensorFIMS->AddComponent(&fieldFIMS);
  sensorFIMS->SetArea(xBoundary[0], yBoundary[0], zBoundary[0], xBoundary[1], yBoundary[1], zBoundary[1]);
  sensorFIMS->AddElectrode(&fieldFIMS, "wtlel");

  //Set up Microscopic Avalanching
  AvalancheMicroscopic* avalancheE = new AvalancheMicroscopic;
  avalancheE->SetSensor(sensorFIMS);
  avalancheE->EnableAvalancheSizeLimit(avalancheLimit);

  ViewDrift viewElectronDrift;
  viewElectronDrift.SetArea(xBoundary[0], yBoundary[0], zBoundary[0], xBoundary[1], yBoundary[1], zBoundary[1]);
  avalancheE->EnablePlotting(&viewElectronDrift, 100);

  //Set up Ion drifting
  AvalancheMC* driftIon = new AvalancheMC;
  driftIon->SetSensor(sensorFIMS);
  driftIon->SetDistanceSteps(MICRONTOCM);

  // ***** Draw field lines for visualization ***** //
  std::cout << "****************************************\n";
  std::cout << "Calculating field Lines.\n";
  std::cout << "****************************************\n";

  DriftLineRKF driftLines(sensorFIMS);
  driftLines.SetMaximumStepSize(MICRONTOCM);

  std::vector<double> xStart;
  std::vector<double> yStart;

  double rangeScale = 0.99;

  //Line generated radially in cardinal directions
  //Note this will make 2*numFieldLines lines
  //    The x-direction is the long axis of the geometry. 
  //    x will extend past the vertex of the hex unit cell
  //    y will go to edge of hex unit cell

  //Field Lines along x:
  for(int i = 0; i < numFieldLine; i++){
    xStart.push_back(rangeScale*xBoundary[1]*i/(numFieldLine-1));
    yStart.push_back(0.);
  }
  //Field Lines along y:
  for(int i = 0; i < numFieldLine; i++){
    xStart.push_back(0.);
    yStart.push_back(rangeScale*yBoundary[1]*i/(numFieldLine-1));
  }

  /*
  //Additional lines radially from center to corner of geometry
  //Adds another numFieldLines lines
  for(int i = 0; i < numFieldLine; i++){
    xStart.push_back(rangeScale*xBoundary[1]*i/(numFieldLine-1));
    yStart.push_back(rangeScale*yBoundary[1]*i/(numFieldLine-1));
  }
  */

  // ***** Calculate field Lines ***** //
  std::vector<std::array<float, 3> > fieldLines;
  int totalFieldLines = xStart.size();
  std::cout << "Computing " << totalFieldLines << " field lines." << std::endl;

  // Note that true number is 3x totalFieldLines - Cathode, above, and below grid.
  int prevDriftLine = 0;
  for(int inFieldLine = 0; inFieldLine < totalFieldLines; inFieldLine++){
    
    fieldLineID = inFieldLine;

    //Calculate from top of volume
    fieldLines.clear();
    driftLines.FieldLine(xStart[inFieldLine], yStart[inFieldLine], zmax*.95, fieldLines);

    //Get coordinates of every point along field line and fill the tree
    for(int inLine = 0; inLine < fieldLines.size(); inLine++){
      fieldLineX = fieldLines[inLine][0];
      fieldLineY = fieldLines[inLine][1];
      fieldLineZ = fieldLines[inLine][2];

      fieldLineDataTree->Fill();
    }

    //Calculate lines from grid - only do those outside of hole
    double lineRadius2 = std::pow(xStart[inFieldLine], 2.) + std::pow(yStart[inFieldLine], 2.);
    double holeRadius2 = std::pow(holeRadius, 2.);
    double gridLineSeparation = 1.1;

    //TODO: The field lines above the grid seem to be causing the simulation to get hung up
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
    if(   (driftLineProgress % 10 == 0)
      &&  (driftLineProgress != prevDriftLine)){
      std::cout << "Driftline Progress: " << driftLineProgress << " %" << std::endl;
      prevDriftLine = driftLineProgress;
    }

  }//End field line loop
  
  std::cout << "Done " << totalFieldLines << " field lines." << std::endl;

  // *** Deal with field line trees *** //

  //Write the field line data tree to file and delete
  //fieldLineDataTree->Print()
  fieldLineDataTree->Write();
  delete fieldLineDataTree;

  //Write the grid field line data tree to file and delete
  //gridFieldLineDataTree->Print()
  gridFieldLineDataTree->Write();
  delete gridFieldLineDataTree;

  // ***** Prepare Avalanche Electron ***** //
  //Set the Initial electron parameters
  double x0 = 0., y0 = 0., z0 = .005;//cm TODO - This is just slightly above grid, but better to parameterize with something
  double t0 = 0.;//ns
  double e0 = 0.1;//eV (Garfield is weird when this is 0.)
  double dx0 = 0., dy0 = 0., dz0 = 0.;//No velocity

  //Start timing the sim
  startSim = clock();
  lapAvalanche = clock();

  std::cout << "****************************************\n";
  std::cout << "Starting simulation: " << runNo << "\n";
  std::cout << "****************************************\n";

  std::cout << "Starting " << numAvalanche << " avalanches." << std::endl;
  //***** Avalanche Loop *****//
  int prevAvalanche = 0;
  for(int inAvalanche = 0; inAvalanche < numAvalanche; inAvalanche++){
    if(DEBUG){
      std::cout << "DEBUGGING - NO AVALANCHE" << std::endl;
      break;
    }
    
    avalancheID = inAvalanche;

    //Reset avalanche data
    totalElectrons = 0;
    attachedElectrons = 0;
    totalIons = 0;
    
    //Begin single-electron avalanche
    avalancheE->AvalancheElectron(x0, y0, z0, t0, e0, dx0, dy0, dz0);

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

      //Drift positive ion from start of every electron track
      ionCharge = 1;
      driftIon->DriftIon(xi, yi, zi, ti);
      driftIon->GetIonEndpoint(0, xiIon, yiIon, ziIon, tiIon, xfIon, yfIon, zfIon, tfIon, statIon);
      //Fill tree with data from this positive ion
      ionDataTree->Fill();
      totalIons++;

      //Check for electron attatchment
      if(stat == -7){
        attachedElectrons++;

        //Drift negative ion from end of electron tracks that attatch
        ionCharge = -1;
        driftIon->DriftNegativeIon(xf, yf, zf, tf);
        driftIon->GetNegativeIonEndpoint(0, xiIon, yiIon, ziIon, tiIon, xfIon, yfIon, zfIon, tfIon, statIon);
        //Fill tree with data from this negative ion
        ionDataTree->Fill();
        totalIons++;
      }

      // Get electron drift line data
      int numElectronDrift = viewElectronDrift.GetNumberOfDriftLines();
      bool isElectron;
      std::vector<std::array<float, 3> > electronDriftLines;
      viewElectronDrift.GetDriftLine(inElectron, electronDriftLines, isElectron);
      for(int inPoint = 0; inPoint < electronDriftLines.size(); inPoint++){
        electronDriftx = electronDriftLines[inPoint][0];
        electronDrifty = electronDriftLines[inPoint][1];
        electronDriftz = electronDriftLines[inPoint][2];

        //Fill tree with data from this point
        electronTrackDataTree->Fill();
      }


      //*** TODO ***/
      //Can insert any per-electron analysis/data here.
      // --Velocity?

      //Fill tree with data from this electron
      electronDataTree->Fill();

    }//end electrons in avalanche loop

    //*** TODO ***/
    //Can insert any per-avalanche analysis/data here.
    // -- Induced signals
    // -- Histograms of energy loss/collison, time between collisions,

    //Fill tree with data from this avalanche
    avalancheDataTree->Fill();

    //Print timing every ~10%
    int avalancheProgress = (100*(inAvalanche+1))/numAvalanche;
    if(  (avalancheProgress % 10 == 0)
      && (avalancheProgress != prevAvalanche)){

      double timeElapsed = (clock() - lapAvalanche) / CLOCKS_PER_SEC;
      lapAvalanche = clock();

      std::stringstream progressStream;
      progressStream << "Done ~" << std::fixed << std::setprecision(0) << avalancheProgress;
      progressStream << "% (~" << std::fixed << std::setprecision(0) << timeElapsed << " s)\n";
      std::cout << progressStream.str() << std::flush;
      prevAvalanche = avalancheProgress;
    }

    //clean up memory
    viewElectronDrift.Clear();

  }//end avalanche loop

  //Final timing
  stopSim = clock();
  runTime = (stopSim - startSim)/CLOCKS_PER_SEC;
  std::cout << "****************************************\n";
  std::cout << "Done processing avalanches...(" << runTime << " s)\n";
  std::cout << "****************************************\n";

  //***** Deal with Root trees and files *****//

  //Fill and write the meta data tree to file then delete
  metaDataTree->Fill();
  //metaDataTree->Print();
  metaDataTree->Write();
  delete metaDataTree;

  //Write the electron data tree to file then delete
  //electronDataTree->Print()
  electronDataTree->Write();
  delete electronDataTree;

  //Write the ion data tree to file then delete
  //ionDataTree->Print()
  ionDataTree->Write();
  delete ionDataTree;

  //Write the avalanche data tree to file then delete
  //avalancheDataTree->Print()
  avalancheDataTree->Write();
  delete avalancheDataTree;

  //Write the electron track data tree to file then delete
  //electronTrackDataTree->Print()
  electronTrackDataTree->Write();
  delete electronTrackDataTree;

  //close the output file
  dataFile->Close();
  delete dataFile;

  std::cout << "****************************************\n";
  std::cout << "Done simulation: " << runNo << "\n";
  std::cout << "****************************************\n";
  std::cout << std::endl;

  return 0;

}
