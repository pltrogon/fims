/*
 * electronAvalanche.cc
 * 
 * Garfield++ simulation of a single-electron avalanche.
 *    Uses garfield analytical components for the field.
 *    Reads simulation parameters from run_control.
 *    Reads a run number from runno.
 *    Saves avalanche data in a .root file as two trees:
 *        metaDataTree
 *        eventDataTree
 * 
 * Tanner Polischuk
 * Last Updated: April 9, 2025
 */

//Garfield includes
#include "Garfield/ComponentAnalyticField.hh"
#include "Garfield/AvalancheMicroscopic.hh"
#include "Garfield/MediumMagboltz.hh"
#include "Garfield/AvalancheMC.hh"
#include "Garfield/Sensor.hh"

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
  clock_t start, stop, lapField, runTime;

  TApplication app("app", &argc, argv);

  //***** Run numbering *****//
  //Read in run number from runno
  int runNo;
  std::string runnoFile = "../runno";
  std::ifstream runInFile;
  runInFile.open(runnoFile);
  if(runInFile.is_open()) {
    runInFile >> runNo;
    runInFile.close();

    std::cout << "****************************************\n";
    std::cout << "Building simulation: " << runNo << "\n";
    std::cout << "****************************************\n";
  }
  else{
      std::cerr << "Error reading file 'runno'." << std::endl;
      return -1;
  }

  //Update runno file
  std::ofstream runOutFile(runnoFile);
  if(runOutFile.is_open()){
    runOutFile << runNo+1;
    runOutFile.close();
  }
  else{
    std::cerr << "Error writing file 'runno'." << std::endl;
    return -1;
  }

  //***** Data File *****//
  std::string dataFilename = "sim."+std::to_string(runNo)+".root";
  std::string dataPath = "../Data/"+dataFilename;
  TFile *dataFile = new TFile(dataPath.c_str(), "NEW");

  if(dataFile->IsOpen()){
      std::cout << "File '" << dataFilename << "' created and opened successfully.\n";
  }else{
    std::cerr << "Error creating or opening file '" << dataFilename << "'." << std::endl;
    return -1;
  }

  //***** Simulation Parameters *****//
  //Read in simulation parameters from run_control
  int numEvent, avalLimit;
  double EField_min, EField_max, EField_step;
  double ar_comp, co2_comp;
  double rPenning, lambdaPenning;
  double standoff, pixelWidth, pitch;

  std::ifstream paramFile;
  paramFile.open("../run_control");
  if(paramFile.is_open()){

    std::cout << "****************************************\n";
    std::cout << "Setting up simulation.\n";
    std::cout << "****************************************\n";
    
    std::string curLine;
    std::map<std::string, std::string> readParam;

    //Read the file contents to a map
    int numKeys = 0;//Number of user-defined simulation parameters to search for is 12.
    while(std::getline(paramFile, curLine)){
      size_t colonPos = curLine.find(":");
      if (colonPos != std::string::npos){
        std::string key = curLine.substr(0, colonPos);
        std::string value = curLine.substr(colonPos + 1);
        readParam[key] = value;
        numKeys++;
      }
    }
    paramFile.close();

    //Parse the values from the map
    if(numKeys == 12){
      numEvent = std::stoi(readParam["Number of simulation events"]);
      avalLimit = std::stoi(readParam["Electron avalanche limit"]);
      EField_min = std::stod(readParam["Minimum E Field (kV/cm)"]);
      EField_max = std::stod(readParam["Maximum E Field (kV/cm)"]);
      EField_step = std::stod(readParam["E Field step (kV/cm)"]);
      ar_comp = std::stod(readParam["Ar gas composition"]);
      co2_comp = std::stod(readParam["CO2 gas composition"]);
      rPenning = std::stod(readParam["Penning transfer (r)"]);
      lambdaPenning = std::stod(readParam["Penning transfer (lambda)"]);
      standoff = std::stod(readParam["Standoff height (micron)"]);
      pixelWidth = std::stod(readParam["Pixel width (micron)"]);
      pitch = std::stod(readParam["Pitch (micron)"]);
    }
    else{
      std::cerr << "Error: Invalid simulation parameters in file 'run_control'." << std::endl;
      return -1;
    }
  }
  else{
    std::cerr << "Error: Could not open simulation parameter file 'run_control'." << std::endl;
    return -1;
  }

  int numEField;

  // ***** Metadata tree ***** //
  //Create
  TTree *metaDataTree = new TTree("metadataTree", "Simulation Parameters");//TODO - change this to metaDataTree

  //Add branches

  //My Running parameters //TODO - These will not be necessary w/ James' geometry
  metaDataTree->Branch("EField_min", &EField_min, "EField_min/D");
  metaDataTree->Branch("EField_max", &EField_max, "EField_max/D");
  metaDataTree->Branch("EField_step", &EField_step, "EField_step/D");
  metaDataTree->Branch("numEField", &numEField, "numEField/I");

  //Geometry Parameters
  metaDataTree->Branch("standoff", &standoff, "standoff/D");
  metaDataTree->Branch("pixelWidth", &pixelWidth, "pixelWidth/D");
  metaDataTree->Branch("pitch", &pitch, "pitch/D");


  //Simulation control parameters
  metaDataTree->Branch("runNo", &runNo, "runNo/I");
  metaDataTree->Branch("numEvent", &numEvent, "numEvent/I");
  metaDataTree->Branch("avalLimit", &avalLimit, "avalLimit/I");

  metaDataTree->Branch("runTime", &runTime, "runTime/D");

  //Simulation component parameters
  metaDataTree->Branch("ar_comp", &ar_comp, "ar_comp/D");
  metaDataTree->Branch("co2_comp", &co2_comp, "co2_comp/D");
  metaDataTree->Branch("rPenning", &rPenning, "rPenning/D");
  metaDataTree->Branch("lambdaPenning", &lambdaPenning, "lambdaPenning/D");



  //*************** SIMULATION ***************//
  std::cout << "****************************************\n";
  std::cout << "Starting simulation: " << runNo << "\n";
  std::cout << "****************************************\n";

  // Define the gas mixture
  MediumMagboltz* gas = new MediumMagboltz();
    gas->SetComposition("ar", ar_comp, "co2", co2_comp);
    gas->SetTemperature(293.15); // Room temperature
    gas->SetPressure(760.);     // Atmospheric pressure
    gas->Initialise(true);
    gas->EnablePenningTransfer(rPenning, lambdaPenning, "ar");

  //Start timing the sim
  start = clock();
  lapField = clock();

  numEField = (EField_max - EField_min)/EField_step + 1;
  
  std::cout << "****************************************\n";
  std::cout << "Processing " << numEField << " electric fields...\n";
  std::cout << "****************************************\n";

  //***** E Field Loop *****//
  //Loop through E field strengths
  for(int inField = 0; inField < numEField; inField++){

    //***** Data Tree *****/
    //Create
    TTree *eventDataTree = new TTree(Form("eventDataTree_EField_%d", inField), Form("Event Simulation Data for E-field %d", inField));


    //Data to be saved for each avalanche
    double EField;
    int totalElectrons;
    int attatchedElectrons;

    //Add Branches
    eventDataTree->Branch("Electric Field", &EField, "EField/D");
    eventDataTree->Branch("Total Electrons", &totalElectrons, "totalElectrons/I");
    eventDataTree->Branch("Attatched Electrons", &attatchedElectrons, "attatchedElectrons/I");

    //Calculate current field and necessary applied voltage
    EField = EField_min + inField*EField_step;
    double voltage = EField*standoff*0.1;//0.1->[cm/micron]*[V/kV]

    //Create an analytic field component
    ComponentAnalyticField* ppField = new ComponentAnalyticField();
      ppField->AddPlaneY(standoff/2e4, -voltage/2.);
      ppField->AddPlaneY(-standoff/2.e4, voltage/2.);
      ppField->SetMedium(gas);
          
    //Create a sensor
    Sensor* sensor = new Sensor();
    sensor->AddComponent(ppField);

    //Set up Microscopic Avalanching
    AvalancheMicroscopic* avalancheE = new AvalancheMicroscopic;
    avalancheE->SetSensor(sensor);
    avalancheE->EnableAvalancheSizeLimit(avalLimit);

    //Set the Initial parameters
    double x0 = 0., y0 = 0.99*standoff/2.e4, z0 = 0.; //cm
    double t0 = 0.;//ns
    double e0 = 0.1;//eV
    double dx0 = 0., dy0 = 0., dz0 = 0.;//

    //***** Avalanche Loop *****//
    //Loop through avalanche events at given field strength
    for(int inEvent = 0; inEvent < numEvent; inEvent++){

      //Update console to indicate running for user
      if((inEvent == 0) && (inField != numEField)){
        std::cout << "." << std::flush;
      }

      //Reset avalanche data
      totalElectrons = 0;
      attatchedElectrons = 0;
      
      //Begin single-electron avalanche
      avalancheE->AvalancheElectron(x0, y0, z0, t0, e0, dx0, dy0, dz0);

      // TODO - Can save each raw avalanche here?

      //Electron count - use endpoints
      int avalancheElectrons = avalancheE->GetNumberOfElectronEndpoints();

      //Extract individual electron data: x,y,z,time,energy.
      double xi, yi, zi, ti, Ei; //Initial
      double xf, yf, zf, tf, Ef; //Final
      int stat; // Electron status

      //Loop through electrons in avalanche
      for(int i = 0; i < avalancheElectrons; i++){
        avalancheE->GetElectronEndpoint(i, xi, yi, zi, ti, Ei, xf, yf, zf, tf, Ef, stat);
          
        totalElectrons++;

        if(stat == -7){
          attatchedElectrons++;
        }

        //*** TODO ***/
        //Can insert any per-electron analysis/data here.
      }//end electrons in avalanche loop

      //*** TODO ***/
      //Can insert any per-avalanche analysis/data here.

      //Fill tree with data from this avalanche event
      eventDataTree->Fill();

    }//end avalanche event loop

    //Write the data tree to file then delete
    //eventDataTree->Print()
    eventDataTree->Write();
    delete eventDataTree; // Clean up the tree object

    //Print timing every ~10%
    if((inField % std::max(1, numEField/10) == 0) && (numEField > 0)){

      double timeElapsed = (clock() - lapField) / CLOCKS_PER_SEC;
      lapField = clock();

      double progress = 100.*(inField+1.)/numEField;
      if(numEField == 1){
        progress = 100.;
      }

      std::cout << "done ~" << std::fixed << std::setprecision(0)
                << progress << "%   (" << timeElapsed << " s)\n";
    }
    
  }//end field strength loop

  //Final timing
  stop = clock();
  runTime = (stop - start)/CLOCKS_PER_SEC;
  std::cout << "****************************************\n";
  std::cout << "Done processing E fields...(" << runTime << " s)\n";
  std::cout << "****************************************\n";
  std::cout << "Done simulation: " << runNo << "\n";
  std::cout << "****************************************\n";

  // ***** Deal with mnetaData ***** //
  metaDataTree->Fill();
  //metaDataTree->Print();
  metaDataTree->Write();
  delete metaDataTree;

  //close the file
  dataFile->Close();
  delete dataFile;

  return 0;

}

