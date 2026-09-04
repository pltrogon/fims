/*
 * avalanche.cc
 * 
 * Garfield++ simulation of a single-electron avalanche.
 *    Requires an input electric field solved by elmer and geometry from gmsh.
 *    Reads simulation parameters from stdin as JSON.
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

//My includes
#include "myFunctions.hh"

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

  //TODO - make this an input or just trial T/F
  bool distOnPlane = true;

  // Handle geometry mode from input
  if(argc != 2){
    std::cerr << "Format: " << argv[0] << " <GeometryMode>" << std::endl;
    return -1;
  }

  std::string geometryModeString = argv[1];
  GeometryMode geometryMode = stringToGeometryMode(argv[1]);
  if(geometryMode == GeometryMode::Unknown){
    std::cerr << "Error: Invalid geometryMode: " << argv[1] << std::endl;
    return -1;
  }

  std::vector<std::string> sensorList;
  sensorList.push_back("CentralPad");
  double cellXScale = 1. / sqrt(3.);
  double cellYScale = 0.5;
  bool hexCell = true;
  
  switch(geometryMode){
    case GeometryMode::Square: {
      // Adjust E-Field line generation points
      cellXScale = 0.5;
      hexCell = false;
      break;
    }
    case GeometryMode::SquareSurrounding: {
      // Add pads to sensor list
      sensorList.push_back("TopPad");
      sensorList.push_back("RightTopPad");
      sensorList.push_back("RightPad");
      sensorList.push_back("RightBottomPad");
      sensorList.push_back("BottomPad");
      sensorList.push_back("LeftBottomPad");
      sensorList.push_back("LeftPad");
      sensorList.push_back("LeftTopPad");
      // Adjust E-Field line generation points
      cellXScale = 0.5;
      hexCell = false;
      break;
    }
    
    case GeometryMode::Hexagonal: {
      // Add pads to sensor list
      sensorList.push_back("RightTopPad");
      break;
    }

    case GeometryMode::HexagonalSurrounding: {
      // Add pads to sensor list
      sensorList.push_back("TopPad");
      sensorList.push_back("RightTopPad");
      sensorList.push_back("RightBottomPad");
      sensorList.push_back("BottomPad");
      sensorList.push_back("LeftBottomPad");
      sensorList.push_back("LeftTopPad");
      break;
    }

    default:
      return -1;
  }
  
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
  TString gitVersion = getGitVersion().c_str();

  //***** Simulation Parameters *****//
  auto simParams = readSimulationParameters();
  if(!simParams){
    return -1;
  }
  int runNo = simParams->runNumber;

  std::cout << "****************************************\n";
  std::cout << "Creating simulation: " << runNo << "\n";
  std::cout << "****************************************\n";


  // ***** Output data file ***** //  
  std::string dataFilename = "sim."+std::to_string(runNo)+".root";
  std::string dataPath = "../../Data/"+dataFilename;
  TFile *dataFile = new TFile(dataPath.c_str(), "RECREATE");
  

  //*************** SIMULATION ***************//
  // Define and initialize the gas mixture
  MediumMagboltz* gasFIMS = initializeGas(*simParams); 

  // Import elmer-generated field map
  std::string geometryPath = "../Geometry/";
  std::string elmerResultsPath = geometryPath+"elmerResults/";
  std::string fieldPath = elmerResultsPath + geometryModeString + ".result";
  ComponentElmer fieldFIMS(
    elmerResultsPath+"mesh.header",
    elmerResultsPath+"mesh.elements",
    elmerResultsPath+"mesh.nodes", 
    geometryPath+"dielectrics.dat",
    fieldPath, 
    "mum"
  );

  // Get region of elmer geometry
  double xmin, ymin, zmin, xmax, ymax, zmax;
  fieldFIMS.GetBoundingBox(xmin, ymin, zmin, xmax, ymax, zmax);

  //Define boundary region for simulation
  double xBoundary[2], yBoundary[2], zBoundary[2];
  zBoundary[0] = zmin;
  zBoundary[1] = zmax;
  //Extend simulation boundary to +/- 2*pitch in x and y
  xBoundary[0] = -2.*simParams->pitch;
  xBoundary[1] = 2.*simParams->pitch;
  yBoundary[0] = -2.*simParams->pitch;
  yBoundary[1] = 2.*simParams->pitch;

  //Enable periodicity and set components
  fieldFIMS.EnableMirrorPeriodicityX();
  fieldFIMS.EnableMirrorPeriodicityY();
  fieldFIMS.SetGas(gasFIMS);

  // Import the weighting field for the readout electrodes.
  for(int i = 0; i < sensorList.size(); i++){
    std::string inSensor = sensorList.at(i);
    std::string fieldFilePath = elmerResultsPath + geometryModeString + inSensor + "Weighting.result";
    fieldFIMS.SetWeightingField(fieldFilePath, inSensor);
  }

  //Create a sensor
  Sensor* sensorFIMS = new Sensor();
  sensorFIMS->AddComponent(&fieldFIMS);
  sensorFIMS->SetArea(
    xBoundary[0], yBoundary[0], zBoundary[0], 
    xBoundary[1], yBoundary[1], zBoundary[1]
  );
  for(const auto& inSensor : sensorList){
    sensorFIMS->AddElectrode(&fieldFIMS, inSensor);
  }
  
  
  // ***** Prepare Avalanche Electron ***** //
  //Set the Initial electron parameters
  double z0 = simParams->initialZFraction * simParams->driftLength;

  double t0 = 0.;//ns
  double e0 = 0.1;//eV (Garfield is weird when this is 0.)
  double dx0 = 0., dy0 = 0., dz0 = 0.;//No velocity

  double timeFinal = 15.;//ns
  double timeStep = 0.01;//ns
  int nSignalBins = timeFinal/timeStep;
  
  //Start timing the sim
  startSim = clock();

  if(simParams->numAvalanche == 0){
    std::cerr << "Warning - No avalanches" << std::endl;

  }
  std::cout << "****************************************\n";
  std::cout << "Starting simulation: " << runNo << "\n";
  std::cout << "****************************************\n";

  std::cout << "Starting " << simParams->numAvalanche << " avalanches." << std::endl;

  //*** Set up parallel avalanche loops ***//
  std::vector<std::string> parallelFileNames;
  #pragma omp parallel
  {
    //thread-local pointers
    ComponentElmer* parallelFieldFIMS = nullptr;
    Sensor* parallelSensorFIMS = nullptr;
    AvalancheMicroscopic* avalancheE = nullptr;
    AvalancheMC* driftIon = nullptr;
    TFile* parallelDataFile = nullptr;
    std::string parallelFilename;

    // Create thread-local objects
    #pragma omp critical
    {//Critical for file I/O
      std::cout << "Setting up for parallel..." << std::endl;

      //Create objects for this thread
      parallelFieldFIMS = new ComponentElmer(
        elmerResultsPath+"mesh.header",
        elmerResultsPath+"mesh.elements",
        elmerResultsPath+"mesh.nodes", 
        geometryPath+"dielectrics.dat",
        fieldPath, 
        "mum"
      );
      parallelSensorFIMS = new Sensor();
      avalancheE = new AvalancheMicroscopic;
      driftIon = new AvalancheMC;

      //Link objects
      parallelFieldFIMS->SetGas(gasFIMS);

      for(int i = 0; i < sensorList.size(); i++){
        std::string inSensor = sensorList.at(i);
        std::string fieldFilePath = elmerResultsPath + geometryModeString + inSensor + "Weighting.result";
        fieldFIMS.SetWeightingField(fieldFilePath, inSensor);
      }
      parallelFieldFIMS->EnableMirrorPeriodicityX();
      parallelFieldFIMS->EnableMirrorPeriodicityY();
      
      parallelSensorFIMS->AddComponent(parallelFieldFIMS);
      parallelSensorFIMS->SetArea(
        xBoundary[0], yBoundary[0], zBoundary[0], 
        xBoundary[1], yBoundary[1], zBoundary[1]
      );      

      for(const auto& inSensor : sensorList){
        parallelSensorFIMS->AddElectrode(&fieldFIMS, inSensor);
      }
      parallelSensorFIMS->SetTimeWindow(0., timeStep, nSignalBins);

      avalancheE->SetSensor(parallelSensorFIMS);
      avalancheE->EnableAvalancheSizeLimit(simParams->avalancheLimit);

      driftIon->SetSensor(parallelSensorFIMS);
      driftIon->SetDistanceSteps(MICRONTOCM);
      driftIon->EnableDriftLines(true);
      
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
    double signalTime; 
    std::vector<double> padSignals(sensorList.size(), 0.0);
    std::vector<double> padSignalSum(sensorList.size(), 0.0);

    TTree* parallelAvalancheDataTree = new TTree("avalancheDataTree", "Avalanche Results");
    parallelAvalancheDataTree->Branch("Avalanche ID", &avalancheID, "avalancheID/I");
    parallelAvalancheDataTree->Branch("Reached Limit", &hitLimit, "hitLimit/B");
    parallelAvalancheDataTree->Branch("Total Electrons", &totalElectrons, "totalElectrons/I");
    parallelAvalancheDataTree->Branch("Attached Electrons", &attachedElectrons, "attachedElectrons/I");
    parallelAvalancheDataTree->Branch("Total Ions", &totalIons, "totalIons/I");
  

    //***** Parallel Avalanche Loop *****//
    #pragma omp for schedule(dynamic)
    for(int inAvalanche = 0; inAvalanche < simParams->numAvalanche; inAvalanche++){
      if(DEBUG){
        continue;
      }
      
      avalancheID = inAvalanche;

      //Reset avalanche data
      totalElectrons = 0;
      attachedElectrons = 0;
      totalIons = 0;

      double cellLength = simParams->pitch*cellXScale;
      auto [x0, y0] = distOnPlane 
        ? randomXYinGeometry(geometryMode, cellLength)
        : std::pair{0.0, 0.0};

      //Begin single-electron avalanche
      avalancheE->AvalancheElectron(x0, y0, z0, 0., e0, dx0, dy0, dz0);

      //Electron count - use endpoints to include attached electrons
      int avalancheElectrons = avalancheE->GetNumberOfElectronEndpoints();

      //Check if avalanche limit was reached
      if(avalancheElectrons >= simParams->avalancheLimit){
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
        
        //Begin extraction of individual ion data
        ionCharge = 1;
        driftIon->DriftIon(xi, yi, zi, ti);
        totalIons++;
        
        //Check for electron attachment
        if(stat == -7){
          attachedElectrons++;

          //Drift negative ion from end of electron tracks that attach
          ionCharge = -1;
          driftIon->DriftNegativeIon(xf, yf, zf, tf);
          totalIons++;
        }


      }//end electrons in avalanche loop

      //*** TODO ***/
      //Can insert any other per-avalanche analysis/data here.
      // -- Histograms of energy loss/collision, time between collisions,

      //Fill tree with data from this avalanche
      parallelAvalancheDataTree->Fill();

    }//end avalanche loop


    // Write and close the file.
    parallelDataFile->Write();
    parallelDataFile->Close();

    delete parallelSensorFIMS;
    delete avalancheE;
    delete driftIon;
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
    simParams->driftField, 0., 0., 1, true,
    vx, vy, driftVelocity, wv, wr, 
    driftDiffusionL, driftDiffusionT,
    alpha, eta, riontof, ratttof, lor, 
    vxerr, vyerr, vzerr, wverr, wrerr, dlerr, dterr,
    alphaerr, etaerr, riontoferr, ratttoferr, lorerr, alphatof,
    difftens
  );

  //Amplification field
  double ampDiffusionL, ampDiffusionT, ampVelocity;
  double ampField = simParams->driftField*simParams->fieldRatio;

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

  metaDataTree->Branch("Pad Length", &simParams->padLength, "padLength/D");
  metaDataTree->Branch("Pitch", &simParams->pitch, "pitch/D");
  metaDataTree->Branch("Amplification Gap", &simParams->amplificationGap, "amplificationGap/D");
  metaDataTree->Branch("Grid Thickness", &simParams->gridThickness, "gridThickness/D");
  metaDataTree->Branch("Pad Thickness", &simParams->padThickness, "padThickness/D");
  metaDataTree->Branch("Hole Radius", &simParams->holeRadius, "holeRadius/D");
  metaDataTree->Branch("Drift Length", &simParams->driftLength, "driftLength/D");
  metaDataTree->Branch("Thickness SiO2", &simParams->thicknessSiO2, "thicknessSiO2/D");
  metaDataTree->Branch("Pillar Radius", &simParams->pillarRadius, "pillarRadius/D");

  metaDataTree->Branch("Electric Field Ratio", &simParams->fieldRatio, "fieldRatio/D");
  metaDataTree->Branch("Drift Field", &simParams->driftField, "driftField/D");
  metaDataTree->Branch("Amplification Field", &ampField, "ampField/D");

  metaDataTree->Branch("Number of Field Lines", &simParams->numFieldLine, "numFieldLine/I");
  metaDataTree->Branch("Number of Avalanches", &simParams->numAvalanche, "numAvalanche/I");
  metaDataTree->Branch("Avalanche Limit", &simParams->avalancheLimit, "avalancheLimit/I");
  metaDataTree->Branch("Initial Z Fraction", &simParams->initialZFraction, "initialZFraction/D");
  
  metaDataTree->Branch("Gas Comp: Ar", &simParams->gasCompAr, "gasCompAr/D");
  metaDataTree->Branch("Gas Comp: CO2", &simParams->gasCompCO2, "gasCompCO2/D");
  metaDataTree->Branch("Gas Comp: CF4", &simParams->gasCompCF4, "gasCompCF4/D");
  metaDataTree->Branch("Gas Comp: Isobutane", &simParams->gasCompIsobutane, "gasCompIsobutane/D");
  metaDataTree->Branch("Gas Penning", &simParams->gasPenning, "gasPenning/D");

  metaDataTree->Branch("Drift Velocity (Drift)", &driftVelocity, "driftVelocity/D");
  metaDataTree->Branch("Diffusion L (Drift)", &driftDiffusionL, "driftDiffusionL/D");
  metaDataTree->Branch("Diffusion T (Drift)", &driftDiffusionT, "driftDiffusionT/D");
  
  metaDataTree->Branch("Drift Velocity (Amplify)", &ampVelocity, "ampVelocity/D");
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
