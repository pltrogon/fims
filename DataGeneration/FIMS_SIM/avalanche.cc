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
 *        avalancheDataTree
 *        electronDataTree
 *        ionDataTree
 * 
 * Tanner Polischuk & James Harrison IV
 */

//Garfield includes
#include "Garfield/ComponentElmer.hh"
#include "Garfield/AvalancheMicroscopic.hh"
#include "Garfield/MediumMagboltz.hh"
#include "Garfield/AvalancheMC.hh"
#include "Garfield/Sensor.hh"
#include "Garfield/ViewFEMesh.hh"
#include "Garfield/ViewField.hh"
#include "Garfield/DriftLineRKF.hh"

//ROOT includes
#include "TApplication.h"
#include "TTree.h"
#include "TFile.h"

//C includes
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <string>
#include <fstream>
#include <iomanip>
#include <map>

using namespace Garfield;

int main(int argc, char * argv[]) {
  
  //*************** SETUP ***************//
  //Timing variables
  clock_t startSim, stopSim, lapAvalanche, runTime;

  TApplication app("app", &argc, argv);

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
  std::string dataPath = "../Data/"+dataFilename;
  TFile *dataFile = new TFile(dataPath.c_str(), "NEW");

  if(!dataFile->IsOpen()){
    std::cerr << "Error creating or opening file '" << dataFilename << "'." << std::endl;
    return -1;
  }  
  
  std::cout << "File '" << dataFilename << "' created and opened successfully.\n";


  //***** Simulation Parameters *****//
  //Read in simulation parameters from runControl

  double  pixelWidth, pixelThickness, pitch;
  double meshStandoff, meshThickness, holeRadius;
  double cathodeHeight, thicknessSiO2;
  double cathodeVoltage, meshVoltage;
  int numFieldLine;
  double transparencyLimit;
  int numAvalanche, avalancheLimit;
  double gasCompAr, gasCompCO2;
  double penningR, penningLambda;

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
  if(numKeys != 18){//Number of user-defined simulation parameters in runControl to search for is 18.
      std::cerr << "Error: Invalid simulation parameters in 'runControl'." << std::endl;
    return -1;
  }

  int debug = 0;
  //Geometry parameters
  pixelWidth = std::stod(readParam["pixelWidth"]);
  pixelThickness = std::stod(readParam["pixelThickness"]);
  pitch = std::stod(readParam["pitch"]);

  meshStandoff = std::stod(readParam["meshStandoff"]);
  meshThickness = std::stod(readParam["meshThickness"]);
  holeRadius = std::stod(readParam["holeRadius"]);

  cathodeHeight = std::stod(readParam["cathodeHeight"]);
  thicknessSiO2 = std::stod(readParam["thicknessSiO2"]);

  //Field parameters
  cathodeVoltage = std::stod(readParam["cathodeVoltage"]);
  meshVoltage = std::stod(readParam["meshVoltage"]);

  numFieldLine = std::stoi(readParam["numFieldLine"]);
  transparencyLimit = std::stoi(readParam["transparencyLimit"]);

  //Simulation Parameters
  numAvalanche = std::stoi(readParam["numAvalanche"]);
  avalancheLimit = std::stoi(readParam["avalancheLimit"]);

  gasCompAr = std::stod(readParam["gasCompAr"]);
  gasCompCO2 = std::stod(readParam["gasCompCO2"]);

  penningR = std::stod(readParam["penningR"]);
  penningLambda = std::stod(readParam["penningLambda"]);


  // ***** Metadata tree ***** //
  //Create
  TTree *metaDataTree = new TTree("metaDataTree", "Simulation Parameters");

  //Data to be saved
  double driftField = (cathodeVoltage - meshVoltage)/cathodeHeight *(10.);// kV/cm
  double amplificationField = meshVoltage/meshStandoff *(10.);// kV/cm
  double fieldTransparency;

  //Add branches
  //General
  metaDataTree->Branch("runNo", &runNo, "runNo/I");

  //Geometry Parameters
  metaDataTree->Branch("Pixel Width", &pixelWidth, "pixelWidth/D");
  metaDataTree->Branch("Pixel Thickness", &pixelThickness, "pixelThickness/D");
  metaDataTree->Branch("Pitch", &pitch, "pitch/D");
  metaDataTree->Branch("Mesh Standoff", &meshStandoff, "meshStandoff/D");
  metaDataTree->Branch("Mesh Thickness", &meshThickness, "meshThickness/D");
  metaDataTree->Branch("Hole Radius", &holeRadius, "holeRadius/D");
  metaDataTree->Branch("Cathode Height", &cathodeHeight, "cathodeHeight/D");
  metaDataTree->Branch("Thickness SiO2", &thicknessSiO2, "thicknessSiO2/D");

  //Field Parameters
  metaDataTree->Branch("Cathode Voltage", &cathodeVoltage, "cathodeVoltage/D");
  metaDataTree->Branch("Mesh Voltage", &meshVoltage, "meshVoltage/D");
  metaDataTree->Branch("Drift Field", &driftField, "driftField/D");
  metaDataTree->Branch("Amplification Field", &amplificationField, "amplificationField/D");
  metaDataTree->Branch("Number of Field Lines", &numFieldLine, "numFieldLine/I");
  metaDataTree->Branch("Field Transparency", &fieldTransparency, "fieldTransparency/D");
  metaDataTree->Branch("Field Transparency Limit", &transparencyLimit, "transparencyLimit/D");

  //Simulation parameters
  metaDataTree->Branch("Number of Avalanches", &numAvalanche, "numAvalanche/I");
  metaDataTree->Branch("Avalanche Limit", &avalancheLimit, "avalancheLimit/I");
  metaDataTree->Branch("Gas Comp: Ar", &gasCompAr, "gasCompAr/D");
  metaDataTree->Branch("Gas Comp: CO2", &gasCompCO2, "gasCompCO2/D");
  metaDataTree->Branch("Penning: r", &penningR, "penningR/D");
  metaDataTree->Branch("Penning: lambda", &penningLambda, "penningLambda/D");
  metaDataTree->Branch("Simulation Run Time", &runTime, "runTime/D");


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

  //***** Avalanche Data Tree *****//
  //Create
  TTree *avalancheDataTree = new TTree("avalancheDataTree", "Avalanche Results");

  //Data to be saved for each avalanche
  int avalancheID;
  bool hitLimit;//T/F if avalanche limit was hit
  int totalElectrons, attatchedElectrons, totalIons;

  //Add Branches
  avalancheDataTree->Branch("Avalanche ID", &avalancheID, "avalancheID/I");
  avalancheDataTree->Branch("Reached Limit", &hitLimit, "hitLimit/B");
  avalancheDataTree->Branch("Total Electrons", &totalElectrons, "totalElectrons/I");
  avalancheDataTree->Branch("Attatched Electrons", &attatchedElectrons, "attatchedElectrons/I");
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


  //*************** SIMULATION ***************//
  std::cout << "****************************************\n";
  std::cout << "Starting simulation: " << runNo << "\n";
  std::cout << "****************************************\n";

  // Define the gas mixture
  MediumMagboltz* gasFIMS = new MediumMagboltz();

  //Set parameters
  gasFIMS->SetComposition("ar", gasCompAr, "co2", gasCompCO2);
  gasFIMS->SetTemperature(293.15); // Room temperature
  gasFIMS->SetPressure(760.);     // Atmospheric pressure
  gasFIMS->Initialise(true);
  // Load the penning transfer and ion mobilities.
  gasFIMS->EnablePenningTransfer(penningR, penningLambda, "ar");
  const std::string path = std::getenv("GARFIELD_INSTALL");
  gasFIMS->LoadIonMobility(path + "/share/Garfield/Data/IonMobility_Ar+_Ar.txt");
  gasFIMS->LoadNegativeIonMobility(path + "/share/Garfield/Data/IonMobility_CO2+_CO2.txt");//TODO - Is this correct for negative ion

  // Import elmer-generated field map
  std::string geometryPath = "../Geometry/";
  ComponentElmer fieldFIMS(geometryPath+"/mesh.header", geometryPath+"/mesh.elements",
                            geometryPath+"/mesh.nodes", geometryPath+"/dielectrics.dat",
                            geometryPath+"/FIMS.result", "cm");

  // Get region of elmer geometry
  double xmin, ymin, zmin, xmax, ymax, zmax;
  fieldFIMS.GetBoundingBox(xmin, ymin, zmin, xmax, ymax, zmax);

  //Enable periodicity and set components
  fieldFIMS.EnablePeriodicityX();
  fieldFIMS.EnablePeriodicityY();
  fieldFIMS.SetGas(gasFIMS);

  // Import the weighting field for the readout electrode.
  //fieldFIMS.SetWeightingField(geometryPath+"FIMS_Weighting.result", "wtlel");//TODO

  //Create a sensor
  Sensor* sensorFIMS = new Sensor();
  sensorFIMS->AddComponent(&fieldFIMS);
  double eps = 1e-6;// To shift coordinates to slightly inside the bounding box
  sensorFIMS->SetArea(-pitch/2.+eps, -pitch/2.+eps, zmin+eps, pitch/2.-eps, pitch/2.-eps, zmax-eps);
  sensorFIMS->AddElectrode(&fieldFIMS, "wtlel");

  //Set up Microscopic Avalanching
  AvalancheMicroscopic* avalancheE = new AvalancheMicroscopic;
  avalancheE->SetSensor(sensorFIMS);
  avalancheE->EnableAvalancheSizeLimit(avalancheLimit);

  //Set up Ion drifting
  AvalancheMC* driftIon = new AvalancheMC;
  driftIon->SetSensor(sensorFIMS);
  driftIon->SetDistanceSteps(1.e-5);

  // ***** Find field transparency ***** //

  std::cout << "****************************************\n";
  std::cout << "Determining field transparency.\n";
  std::cout << "****************************************\n";

  DriftLineRKF driftLines(sensorFIMS);

  //Create 2D linearly-spaced array at top of drift volume
  std::vector<double> xStart;
  std::vector<double> yStart;
  std::vector<double> zStart;

  /*
  for(int i = 0; i < numFieldLine; i++){
    for(int j = 0; j < numFieldLine; j++){
      xStart.push_back(-pitch/2. + (pitch*i/(numFieldLine - 1)));
      yStart.push_back(-pitch/2. + (pitch*j/(numFieldLine - 1)));
      zStart.push_back(zmax-eps);// Shift to slightly inside of volume
    }
  }
  */

  for(int i = 1; i < numFieldLine-1; i++){
    xStart.push_back(-pitch/2. + (pitch*i/(numFieldLine - 1)));
    yStart.push_back(0.);
    zStart.push_back(zmax-eps);
  }


  //Calculate field Lines
  std::vector<std::array<float, 3> > fieldLines;
  int totalFieldLines = xStart.size();
  int numAtPixel = 0;
  std::cout << "Comuputing " << totalFieldLines << " field lines." << std::endl;
  for(int inFieldLine = 0; inFieldLine < totalFieldLines; inFieldLine++){
    fieldLineID = inFieldLine;

    driftLines.FieldLine(xStart[inFieldLine], yStart[inFieldLine], zStart[inFieldLine], fieldLines);

    //Get coordinates of every point along field line and fill the tree
    for(int inLine = 0; inLine < fieldLines.size(); inLine++){
      fieldLineX = fieldLines[inLine][0];
      fieldLineY = fieldLines[inLine][1];
      fieldLineZ = fieldLines[inLine][2];

      fieldLineDataTree->Fill();
    }

    //Find if termination point is at pixel
    int lineEnd = fieldLines.size() - 1;

    std::cout << "Line end: (" << fieldLines[lineEnd][0] << ", " << fieldLines[lineEnd][1] << ", " << fieldLines[lineEnd][2] << ")\n";

    if(  (abs(fieldLines[lineEnd][0]) <= pitch/2.)
      && (abs(fieldLines[lineEnd][1]) <= pitch/2.)
      && (abs(fieldLines[lineEnd][2]) < meshStandoff)//TODO - meshStandoff is not a good criteria here
      ){ 
        numAtPixel++;
      }
  }
  
  //Determine transparency
  fieldTransparency = (1.*numAtPixel) / (1.*totalFieldLines);
  if(fieldTransparency < transparencyLimit){
    std::cerr << "Field transparency is " << fieldTransparency <<  ". (Run " << runNo << ")" << std::endl;
    return -1;
  }
  std::cout << "numAtPixel = " << numAtPixel << "\n";
  std::cout << "totalFieldLines = " << totalFieldLines << "\n";
  std::cout << "Limit: " << transparencyLimit*100. << "\n";
  std::cout << "Transparency: " << fieldTransparency*100. << "\n";


  // ***** Prepare Simulation ***** //
  //Set the Initial electron parameters
  double x0 = 0., y0 = 0., z0 = meshStandoff*1.1; //cm
  double t0 = 0.;//ns
  double e0 = 0.1;//eV (Garfield is weird when this is 0.)
  double dx0 = 0., dy0 = 0., dz0 = 0.;//No velocity

  //Start timing the sim
  startSim = clock();
  lapAvalanche = clock();
  std::cout << "Starting " << numAvalanche << " simulations.\n" << std::flush;

  //***** Avalanche Loop *****//
  for(int inAvalanche = 0; inAvalanche < numAvalanche; inAvalanche++){
    avalancheID = inAvalanche;

    //Reset avalanche data
    totalElectrons = 0;
    attatchedElectrons = 0;
    totalIons = 0;
    
    std::cout << "----------------------\n";
    std::cout << "Starting Avalanche\n";
    //Begin single-electron avalanche
    avalancheE->AvalancheElectron(x0, y0, z0, t0, e0, dx0, dy0, dz0);
    std::cout << "Avalanche Complete\n";
    std::cout << "----------------------\n";

    //Electron count - use endpoints to include attatched electrons
    int avalancheElectrons = avalancheE->GetNumberOfElectronEndpoints();

    //Check if avalanche limit was reached - TODO Decide if this data is worth saving or not
    if(avalancheElectrons >= avalancheLimit){
      hitLimit = true;
    }
    else{
      hitLimit = false;
    }

    //Loop through allelectrons in avalanche
    for(int inElectron = 0; inElectron < avalancheElectrons; inElectron++){
      electronID = inElectron;

      //Extract individual electron data
      avalancheE->GetElectronEndpoint(inElectron, xi, yi, zi, ti, Ei, xf, yf, zf, tf, Ef, stat);
        
      totalElectrons++;

      //Drift positive ion from start of every electron track
      ionCharge = 1;
      driftIon->DriftIon(xi, yi, zi, ti);
      driftIon->GetIonEndpoint(totalIons, xiIon, yiIon, ziIon, tiIon, xfIon, yfIon, zfIon, tfIon, statIon);
      totalIons++;
      //Fill tree with data from this positive ion
      ionDataTree->Fill();

      //Check for electron attatchment
      if(stat == -7){
        attatchedElectrons++;

        //Drift negative ion from end of electron tracks that attatch
        ionCharge = -1;
        driftIon->DriftNegativeIon(xf, yf, zf, tf);
        driftIon->GetIonEndpoint(totalIons, xiIon, yiIon, ziIon, tiIon, xfIon, yfIon, zfIon, tfIon, statIon);
        totalIons++;
        //Fill tree with data from this negative ion
        ionDataTree->Fill();
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
    // -- Histograms of energy loss/collison, ionization posiitions, time between collisions,

    //Fill tree with data from this avalanche
    avalancheDataTree->Fill();


    //Print timing every ~10%
    if(inAvalanche % (numAvalanche/10) == 0){

      double timeElapsed = (clock() - lapAvalanche) / CLOCKS_PER_SEC;
      lapAvalanche = clock();

      double progress = 100.*(inAvalanche+1.)/numAvalanche;

      std::stringstream progressStream;
      progressStream << "Done ~ " << std::fixed << std::setprecision(1) << progress;
      progressStream << "% (" << std::fixed << std::setprecision(2) << timeElapsed << ")\n";
      std::cout << progressStream.str() << std::flush;
    }

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

  //Write the field line data tree to file and delete
  //fieldLineDataTree->Print()
  fieldLineDataTree->Write();
  delete fieldLineDataTree;

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

  //close the output file
  dataFile->Close();
  delete dataFile;

  std::cout << "****************************************\n";
  std::cout << "Done simulation: " << runNo << "\n";
  std::cout << "****************************************\n";

  return 0;

}
