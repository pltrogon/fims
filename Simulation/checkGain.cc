/*
 * checkGain.cc
 *
 * TODO
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


  const double MICRON = 1e-6;
  const double CM = 1e-2;
  const double MICRONTOCM = 1e-4;
  bool DEBUG = false;
  
  //*************** SETUP ***************//
  //Timing variables
  clock_t startSim, stopSim, runTime;

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
  std::cout << "Building simulation: " << runNo << " (gain)\n";
  std::cout << "****************************************\n";

  //***** Simulation Parameters *****//
  //Read in simulation parameters from runControl

  double  padLength, pitch;
  double gridStandoff, gridThickness, holeRadius;
  double cathodeHeight, thicknessSiO2, pillarRadius;
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
  if(numKeys != 15){//Number of user-defined simulation parameters in runControl to search for.
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
  fieldRatio = std::stod(readParam["fieldRatio"]);
  transparencyLimit = std::stod(readParam["transparencyLimit"]);

  numFieldLine = std::stoi(readParam["numFieldLine"]);

  //Simulation Parameters
  numAvalanche = std::stoi(readParam["numAvalanche"]);
  avalancheLimit = std::stoi(readParam["avalancheLimit"]);

  gasCompAr = std::stod(readParam["gasCompAr"]);
  gasCompCO2 = std::stod(readParam["gasCompCO2"]);

  
  //*************** SIMULATION ***************//
  std::cout << "****************************************\n";
  std::cout << "Creating simulation: " << runNo << " (gain)\n";
  std::cout << "****************************************\n";

  // Define the gas mixture
  MediumMagboltz* gasFIMS = new MediumMagboltz();

  //Set parameters
  gasFIMS->SetComposition(
    "ar", gasCompAr, 
    "co2", gasCompCO2
  );

  //gas parameters:
  double gasTemperature = 293.15; //K
  double gasPressure = 760.;//torr
  int maxElectronE = 200;
  double rPenning = 0.51;

  gasFIMS->SetTemperature(gasTemperature);
  gasFIMS->SetPressure(gasPressure);
  gasFIMS->SetMaxElectronEnergy(maxElectronE);
  gasFIMS->Initialise(true);
  // Load the penning transfer and ion mobilities.
  gasFIMS->EnablePenningTransfer(rPenning, .0, "ar");

  const std::string path = std::getenv("GARFIELD_INSTALL");
  const std::string posIonPath = path + "/share/Garfield/Data/IonMobility_Ar+_Ar.txt";
  const std::string negIonPath = path + "/share/Garfield/Data/IonMobility_CO2+_CO2.txt";
  gasFIMS->LoadIonMobility(posIonPath);
  gasFIMS->LoadNegativeIonMobility(negIonPath);//TODO - Is this correct for negative ion

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

  //Create a sensor
  Sensor* sensorFIMS = new Sensor();
  sensorFIMS->AddComponent(&fieldFIMS);
  sensorFIMS->SetArea(
    xBoundary[0], yBoundary[0], zBoundary[0], 
    xBoundary[1], yBoundary[1], zBoundary[1]
  );

  AvalancheMicroscopic* avalancheE = new AvalancheMicroscopic;
	avalancheE->SetSensor(sensorFIMS);
  avalancheE->EnableAvalancheSizeLimit(avalancheLimit);
      
  // ***** Prepare Avalanche Electron ***** //
  //Set the Initial electron parameters
  double x0 = 0., y0 = 0., z0 = holeRadius;
  double t0 = 0.;//ns
  double e0 = 0.1;//eV (Garfield is weird when this is 0.)
  double dx0 = 0., dy0 = 0., dz0 = 0.;//No velocity
  
  //Start timing the sim
  startSim = clock();

  std::cout << "****************************************\n";
  std::cout << "Starting simulation: " << runNo << " (gain)\n";
  std::cout << "****************************************\n";

  //Set up variables for simulation
  int runLimit = 10000;
  int totalAvalanches = 0;
  int numHitLimit = 0;
	int numValid = 0;
	double fractionValid = 0.;

  double gainSum = 0.;
  double sumSquares = 0.;

  double curMean = 0.;
  double curStdDev = 0.;
  double variance = 0.;
  double stdErrMean = 0.;
	double convergeThreshold = 0.2;

	int numInBunch = 100;//Always do at least 100 avalanches

  //Begin simulating electron avalanches until gain is known to convergence threshold
  bool knownGain = false;
  while(!knownGain && totalAvalanches < runLimit){
    
    if(DEBUG){
        break;
      }


    //Do bunch of avalanches
    for(int inAvalanche = 0; inAvalanche < numInBunch; inAvalanche++){
      
      totalAvalanches++;

      //Begin single-electron avalanche
      avalancheE->AvalancheElectron(x0, y0, z0, 0., e0, dx0, dy0, dz0);

      //Electron count - use endpoints to include attached electrons
      int avalancheElectrons = avalancheE->GetNumberOfElectronEndpoints();

      //Check if avalanche limit was reached - do not include these
      if(avalancheElectrons >= avalancheLimit){
        numHitLimit++;
        continue;
      }

      //Increment stats counters
      gainSum += avalancheElectrons;
      sumSquares += avalancheElectrons*avalancheElectrons;

    }//end of avalanche bunch loop

		numInBunch = 10;//do bunches of 10

    numValid = totalAvalanches - numHitLimit;
		fractionValid = numValid / totalAvalanches;
    if(numValid < 2 || fractionValid < .9){
      std::cerr << "Error: Overflow of avalanche limit." << std::endl;
      break;
    }

    //Find new mean gain
    curMean = gainSum / numValid;
    if(curMean < 5){
      std::cerr << "Error: Gain is low." << std::endl;
      break;
    }

    //find std dev
    variance = (sumSquares - gainSum*curMean) / (numValid-1);

    if(variance < 0){
      variance = 0;
    }
    curStdDev = std::sqrt(variance);

    //standard error of mean
    stdErrMean = curStdDev / std::sqrt(numValid);

		//Check if gain has stabilized
		if(stdErrMean < convergeThreshold*curMean){
			knownGain = true;
		}

		//occasionaly print values
		if(totalAvalanches%100 == 0 || knownGain){
			std::cout << "Total valid runs: " << numValid << " (of " << totalAvalanches << ")\n";
    	std::cout << "\tMean gain: " << curMean << " +/- " << stdErrMean << std::endl;
		}
    

  }//end gain convergence loop
  

	//***** Output gain value *****//	
	//create output file
  std::string dataFilename = "gainFile.txt";
  std::string dataPath = "../../Data/"+dataFilename;
	std::ofstream dataFile;

	//Write results to file
	dataFile.open(dataPath);

	//write some extra information
	dataFile << "// Finding gain for run: " << runNo << "\n";
	dataFile << "// Run limit: " << runLimit << "\n";
	dataFile << "// Total valid runs: " << numValid << " (of " << totalAvalanches << ")\n";
	
	if(knownGain){	//write gain if converged
	 	dataFile << curMean << std::endl;
		dataFile << stdErrMean << std::endl;
	}
	else if(totalAvalanches >= runLimit){	//If did not converge, write values as -1
		dataFile << -1 << std::endl;
		dataFile << -1 << std::endl;
  }
	else{ // write -2 for any other cases
		dataFile << -2 << std::endl;
		dataFile << -2 << std::endl;
	}
	dataFile.close();


  //Final timing
  stopSim = clock();
  runTime = (stopSim - startSim)/CLOCKS_PER_SEC;
  std::cout << "****************************************\n";
  std::cout << "Done processing avalanches...(" << runTime << " s)\n";
  std::cout << "****************************************\n";

  std::cout << "****************************************\n";
  std::cout << "Done simulation: " << runNo << " (gain)\n";
  std::cout << "****************************************\n";
  std::cout << std::endl;

  return 0;

}
