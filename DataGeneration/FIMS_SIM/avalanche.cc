/*
 * avalanche.cc
 * 
 * Garfield++ simulation of a single-electron avalanche.
 *    Requires an input electric field solved by elmer and geometry from gmsh.
 *    Reads simulation parameters from run_control.
 *    Reads a run number from runno.
 *    Saves avalanche data in a .root file as root trees:
 *        metaDataTree
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
  clock_t start, stop, lapAvalanche, runTime;

  TApplication app("app", &argc, argv);

  //***** Run numbering *****//
  //Read in run number from runno
  int runNo;
  std::string runnoFile = "../runno";
  std::ifstream runInFile;
  runInFile.open(runnoFile);

  if(!runInFile.is_open()){
    std::cerr << "Error reading file ' << runnnoFile << "'. << std::endl;
    return -1;
  }

  runInFile >> runNo;
  runInFile.close();

  std::cout << "****************************************\n";
  std::cout << "Building simulation: " << runNo << "\n";
  std::cout << "****************************************\n";

  //Update runno file
  std::ofstream runOutFile(runnoFile);
  if(!runOutFile.is_open()){
    std::cerr << "Error writing file ' << runnnoFile << "'. << std::endl;
    return -1;
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
  //Read in simulation parameters from run_control
  int numEvent, avalLimit;
  double ar_comp, co2_comp;
  double rPenning, lambdaPenning;
  double standoff, driftHeight;
  double holeRadius, pixelWidth, pitch;

  std::ifstream paramFile;
  paramFile.open("../run_control");

  if(!paramFile.is_open()){
    std::cerr << "Error: Could not open simulation parameter file 'run_control'." << std::endl;
    return -1;
  }

  std::cout << "****************************************\n";
  std::cout << "Setting up simulation.\n";
  std::cout << "****************************************\n";
  
  std::string curLine;
  std::map<std::string, std::string> readParam;

  //Read the file contents to a map
  int numKeys = 0;//Number of user-defined simulation parameters in run_control to search for is 13.
  while(std::getline(paramFile, curLine)){
    if(curLine.find('#') == 0){
      continue;
    }

    size_t keyPos = curLine.find("=");
    //This assumes that the form "key = value;" Note single whitespace on either side of '='.
    if (keyPos != std::string::npos){
      std::string key = curLine.substr(0, keyPos - 2);
      std::string value = curLine.substr(keyPos + 2);
      if(value.back() == ";"){
        value.pop_back();
      }

      readParam[key] = value;
      numKeys++;
    }
  }
  paramFile.close();

  //Parse the values from the map
  if(numKeys != 13){
      std::cerr << "Error: Invalid simulation parameters in file 'run_control'." << std::endl;
    return -1;
  }

  numEvent = std::stoi(readParam["simulation_Events"]);
  avalLimit = std::stoi(readParam["avalanche_Limit"]);

  ar_comp = std::stod(readParam["ar_comp"]);
  co2_comp = std::stod(readParam["co2_comp"]);
  rPenning = std::stod(readParam["penning_r"]);
  lambdaPenning = std::stod(readParam["penning_lambda"]);

  standoff = std::stod(readParam["standoff_height"]);
  driftHeight = std::stod(readParam["drift_height"]);
  meshTickness = std::stod(readParam["thickness_mesh"]);
  pixelTickness = std::stod(readParam["thickness_pixel"]);
  holeRadius = std::stod(readParam["radius_hole"]);
  pixelWidth = std::stod(readParam["pixel_width"]);
  pitch = std::stod(readParam["pitch"]);

  // ***** Metadata tree ***** //
  //Create
  TTree *metaDataTree = new TTree("metaDataTree", "Simulation Parameters");

  //Add branches
  //Simulation control parameters
  metaDataTree->Branch("runNo", &runNo, "runNo/I");
  metaDataTree->Branch("numEvent", &numEvent, "numEvent/I");
  metaDataTree->Branch("avalLimit", &avalLimit, "avalLimit/I");

  //Geometry Parameters
  //FORMAT: metaDataTree->Branch("", &, "/D");
  metaDataTree->Branch("standoff", &standoff, "standoff/D");
  metaDataTree->Branch("pixelWidth", &pixelWidth, "pixelWidth/D");
  metaDataTree->Branch("pitch", &pitch, "pitch/D");

  //Simulation component parameters
  metaDataTree->Branch("ar_comp", &ar_comp, "ar_comp/D");
  metaDataTree->Branch("co2_comp", &co2_comp, "co2_comp/D");
  metaDataTree->Branch("rPenning", &rPenning, "rPenning/D");
  metaDataTree->Branch("lambdaPenning", &lambdaPenning, "lambdaPenning/D");

  //Other
  metaDataTree->Branch("runTime", &runTime, "runtTime/D");

  //***** Avalanche Data Tree *****//

  //Create
  TTree *avalancheDataTree = new TTree("avalancheDataTree", "Avalanche Results");

  //Data to be saved for each avalanche
  int aval_ID;
  bool hitLimit;//T/F if avalanche limit was hit
  int totalElectrons, attatchedElelectrons, totalIons;

  //Add Branches
  avalancheDataTree->Branch("Avalanche ID", &aval_ID, "aval_ID/I");
  avalancheDataTree->Branch("Reached Limit", &hitLimit, "hitLimit/B");
  avalancheDataTree->Branch("Total Electrons", &totalElectrons, "totalElectrons/I");
  avalancheDataTree->Branch("Attatched Electrons", &attatchedElectrons, "attatchedElectrons/I");
  avalancheDataTree->Branch("Total Ions", &totalIons, "totalIons/I");


  //***** Electron Data Tree *****//

  //Create
  TTree *electronDataTree = new TTree("electronDataTree", "Avalanche Electron Parameters");

  // Data to be saved for each electron
  int electron_ID;
  double xi, yi, zi, ti, Ei; //Initial parameters
  double xf, yf, zf, tf, Ef; //Final parameters
  int stat; // Electron status

  //Add Branches
  electronDataTree->Branch("Avalanche ID", &aval_ID, "aval_ID/I");
  electronDataTree->Branch("Electron ID", &electron_ID, "electron_ID/I");

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
  double xi_ion, yi_ion, zi_ion, ti_ion, Ei_ion; //Initial parameters
  double xf_ion, yf_ion, zf_ion, tf_ion, Ef_ion; //Final parameters
  int stat_ion;

  //Add Branches
  ionDataTree->Branch("Avalanche ID", &aval_ID, "aval_ID/I");
  ionDataTree->Branch("Electron ID", &electron_ID, "electron_ID/I");
  ionDataTree->Branch("Ion Charge", &ionCharge, "ionCharge/I");

  ionDataTree->Branch("Initial x", &xi_ion, "xi_ion/D");
  ionDataTree->Branch("Initial y", &yi_ion, "yi_ion/D");
  ionDataTree->Branch("Initial z", &zi_ion, "zi_ion/D");
  ionDataTree->Branch("Initial Time", &ti_ion, "ti_ion/D");
  ionDataTree->Branch("Initial Energy", &Ei_ion, "Ei_ion/D");

  ionDataTree->Branch("Final x", &xf_ion, "xf_ion/D");
  ionDataTree->Branch("Final y", &yf_ion, "yf_ion/D");
  ionDataTree->Branch("Final z", &zf_ion, "zf_ion/D");
  ionDataTree->Branch("Final Time", &tf_ion, "tf_ion/D");
  ionDataTree->Branch("Final Energy", &Ef_ion, "Ef_ion/D");

  ionDataTree->Branch("Exit Status", &stat_ion, "stat_ion/I");


  //*************** SIMULATION ***************//
  std::cout << "****************************************\n";
  std::cout << "Starting simulation: " << runNo << "\n";
  std::cout << "****************************************\n";

  // Define the gas mixture
  MediumMagboltz* gas = new MediumMagboltz();

  //Set parameters
  gas->SetComposition("ar", ar_comp, "co2", co2_comp);
  gas->SetTemperature(293.15); // Room temperature
  gas->SetPressure(760.);     // Atmospheric pressure
  gas->Initialise(true);
  // Load the penning transfer and ion mobilities.
  gas->EnablePenningTransfer(rPenning, lambdaPenning, "ar");
  const std::string path = std::getenv("GARFIELD_INSTALL");
  gas->LoadIonMobility(path + "/share/Garfield/Data/IonMobility_Ar+_Ar.txt");
  gas->LoadNegativeIonMobility(path + "/share/Garfield/Data/IonMobility_CO2+_CO2.txt");//TODO - Is this correct for negative ion

  // Import elmer-generated field map
  std::string geometryPath = "../Geometry/";
  ComponentElmer FIMS_Field(geometryPath+"/mesh.header", geometryPath+"/mesh.elements",
                            geometryPath+"/mesh.nodes", geometryPath+"/dielectrics.dat",
                            geometryPath+"/FIMS.result", "cm");

  FIMS_Field.EnablePeriodicityX();
  FIMS_Field.EnablePeriodicityY();
  FIMS_Field.SetGas(&gas);

  // Import the weighting field for the readout electrode.
  FIMS_Field.SetWeightingField(geometryPath+"FIMS_Weighting.result", "wtlel");

  //Create a sensor
  Sensor* sensor = new Sensor();
  sensor->AddComponent(&FIMS_Field);
  sensor->SetArea(-pitch/2., -pitch/2., TODO_NEGATIVEZLIMIT, pitch/2., pitch/2., TODO_POSITIVEZLIMIT)
  sensor->AddElectrode(&FIMS_Field, "wtlel")

  //Set up Microscopic Avalanching
  AvalancheMicroscopic* avalancheE = new AvalancheMicroscopic;
  avalancheE->SetSensor(sensor);
  avalancheE->EnableAvalancheSizeLimit(avalLimit);

  //Set up Ion drifting
  AvalancheMC* driftIon = new AvalancheMC;
  driftIon->SetSensor(sensor);
  driftIon->SetDistanceSteps(1.e-5);

  //***********************************************************************************************************************
  //***********************************************************************************************************************
  //***********************************************************************************************************************
  //***********************************************************************************************************************
  //***********************************************************************************************************************
  //***********************************************************************************************************************
  DriftLineRKF driftLines(&sensor);

  //calculate field lines for visualization
  std::vector<double> xStart;
  std::vector<double> yStart;
  std::vector<double> zStart;
  int nLines = 23;
  for (int i = 0; i < nLines; i++) {
    xStart.push_back(-pitch/2. + (pitch*i/(nLines - 1)));
    yStart.push_back(0.);
    zStart.push_back(0.);//TODO -  top of drifft volume
  }
  
  TCanvas * fieldLines = new TCanvas("geometryFieldLines", "Field Line Geometry");
  
  ViewFEMesh viewGeometry(&FIMS_Field);
  viewGeometry.SetArea();
  viewGeometry.SetCanvas(fieldLines);
  viewGeometry.SetPlane(0, -1, 0, 0, 0, 0);
  viewGeometry.SetFillMesh(true);




  

  //E field Lines
  TCanvas* fieldLines = new TCanvas("fieldLines", "", 600, 600);
  viewField.SetCanvas(fieldLines);
  viewField.PlotFieldLines(xStart, yStart, zStart);
  pixel1D->Draw();
  gPad->Update();

  //Weighting Field for pixel
  TCanvas* weightingField = new TCanvas("weightingField", "", 600, 600);
  viewField.SetCanvas(weightingField);
  viewField.PlotContourWeightingField("pixel", "v");
  pixel1D->Draw();
  gPad->Update();



  TCanvas * fieldLines = new TCanvas("geometryFieldLines", "Field Line Geometry");
  const bool plotContours = false;


  //Mesh
  viewFEMesh = 


  // ***** Prepare Simulation ***** //
  //Set the Initial electron parameters
  double x0 = 0., y0 = 0., z0 = 0.; //cm
  double t0 = 0.;//ns
  double e0 = 0.1;//eV (Garfield is weird when this is 0.)
  double dx0 = 0., dy0 = 0., dz0 = 0.;//No velocity

  //Start timing the sim
  start = clock();
  lapField = clock();
  std::cout << "Starting " << numEvent << " simulation events." << std::flush;

  //***** Avalanche Loop *****//
  for(int inEvent = 0; inEvent < numEvent; inEvent++){
    aval_ID = inEvent;

    //Reset avalanche data
    totalElectrons = 0;
    attatchedElectrons = 0;
    totalIons = 0;
    
    //Begin single-electron avalanche
    avalancheE->AvalancheElectron(x0, y0, z0, t0, e0, dx0, dy0, dz0);

    //Electron count - use endpoints to include attatched electrons
    int avalancheElectrons = avalancheE->GetNumberOfElectronEndpoints();

    //Check if avalanche limit was reached - TODO Decide if this data is worth saving or not
    if(avalancheElectrons >= avalLimit){
      hitLimit = true;
    }
    else{
      hitLimit = false;
    }

    //Loop through electrons in avalanche
    for(int i = 0; i < avalancheElectrons; i++){
      electron_ID = i;

      //Extract individual electron data
      avalancheE->GetElectronEndpoint(i, xi, yi, zi, ti, Ei, xf, yf, zf, tf, Ef, stat);
        
      totalElectrons++;

      //Drift positive ion from start of every electron track
      ionCharge = 1;
      driftIon->DriftIon(xi, yi, zi, ti);
      driftIon->GetIonEndpoint(totalIons, xi_ion, yi_ion, zi_ion, ti_ion, Ei_ion, xf_ion, yf_ion, zf_ion, tf_ion, Ef_ion, stat_ion)
      totalIons++;
      //Fill tree with data from this positive ion
      ionDataTree->Fill();

      //Check for electron attatchment
      if(stat == -7){
        attatchedElectrons++;

        //Drift negative ion from end of electron tracks that attatch
        ionCharge = -1;
        driftIon->DriftNegativeIon(xf, yf, zf, tf);
        driftIon->GetIonEndpoint(totalIons, xi_ion, yi_ion, zi_ion, ti_ion, Ei_ion, xf_ion, yf_ion, zf_ion, tf_ion, Ef_ion, stat_ion)
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
    if(inEvent % numEvent/10 == 0){

      double timeElapsed = (clock() - lapAvalanche) / CLOCKS_PER_SEC;
      lapAvalanche = clock();

      double progress = 100.*(inEvent+1.)/numEvent;

      std::cout << fmt::format("done ~ {}% ({})\n", progress, timeElapsed) << std::flush;
    }

  }//end avalanche event loop

  //Final timing
  stop = clock();
  runTime = (stop - start)/CLOCKS_PER_SEC;
  std::cout << "****************************************\n";
  std::cout << "Done processing events...(" << runTime << " s)\n";
  std::cout << "****************************************\n";

  //***** Deal with Root trees and files *****//

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

  //Write the meta data tree to file then delete
  metaDataTree->Fill();
  //metaDataTree->Print();
  metaDataTree->Write();
  delete metaDataTree;

  //close the output file
  dataFile->Close();
  delete dataFile;

  std::cout << "****************************************\n";
  std::cout << "Done simulation: " << runNo << "\n";
  std::cout << "****************************************\n";

  return 0;

}
