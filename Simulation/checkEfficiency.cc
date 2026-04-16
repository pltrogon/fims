/*
 * checkEfficiency.cc
 *
 * 
 * Garfield++ simulation of single-electron avalanches in a FIMS geometry.
 * 
 * Repeats avalanches until an efficiency of 95% with a 10-electron threshold 
 * is met or excluded with a 2-signma confidence.
 * 
 * Input parameters are:
 * <Target Efficiency>
 * <Efficiency Threshold>
 * 
 * Results are written to a file: "efficiencyFile.dat"
 * 
 * Tanner Polischuk & James Harrison IV
 * December 2025
 */

// My includes
#include "SilenceConsole.h"
#include "myFunctions.h"

//Garfield includes
#include "Garfield/ComponentElmer.hh"
#include "Garfield/AvalancheMicroscopic.hh"
#include "Garfield/MediumMagboltz.hh"
#include "Garfield/Medium.hh"
#include "Garfield/AvalancheMC.hh"
#include "Garfield/Sensor.hh"

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

  if(argc != 3){
    std::cerr << "Format: " << argv[0] << " <Target Efficiency> <Threshold>" << std::endl;
    return 1;
  }

  int electronThreshold = std::atoi(argv[2]);
  double targetEfficiency = std::atof(argv[1]);

  const double confidenceValue = 2;//NOTE - 1.645 for 95% confidence instead of 2 for 2-sigma

  //Random seed
  std::srand(static_cast<unsigned int>(std::time(nullptr)));

  const double MICRON = 1e-6;
  const double CM = 1e-2;
  const double MICRONTOCM = 1e-4;
  
  //*************** SETUP ***************//
  //Timing variables
  clock_t startSim, stopSim, runTime;

  //***** Simulation Parameters *****//
  //Read in simulation parameters from stdin as JSON
  std::cout << "****************************************\n";
  std::cout << "Setting up simulation.\n";
  std::cout << "****************************************\n";

  // Read JSON parameters from stdin
  auto simParams = readSimulationParameters();
  if(!simParams){
    return -1;
  }
  int runNo = simParams->runNumber;

  std::cout << "****************************************\n";
  std::cout << "Building simulation: " << runNo << " (efficiency)\n";
  std::cout << "****************************************\n";

  //*************** SIMULATION ***************//
  std::cout << "****************************************\n";
  std::cout << "Creating simulation: " << runNo << " (efficiency)\n";
  std::cout << "****************************************\n";

  // Define and initialize the gas mixture
  MediumMagboltz* gasFIMS = initializeGas(*simParams);

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
  //TODO - This may need to be much larger
  xBoundary[0] = -simParams->pitch;
  xBoundary[1] = simParams->pitch;
  yBoundary[0] = -simParams->pitch;
  yBoundary[1] = simParams->pitch;

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
  avalancheE->EnableAvalancheSizeLimit(simParams->avalancheLimit);
      
  // ***** Prepare Avalanche Electron ***** //
  //Set the Initial electron parameters
  double x0 = 0., y0 = 0., z0 = simParams->holeRadius;//x0 = pitch/std::sqrt(3)
  double t0 = 0.;//ns
  double e0 = 0.1;//eV (Garfield is weird when this is 0.)
  double dx0 = 0., dy0 = 0., dz0 = 0.;//No velocity
  
  //Start timing the sim
  startSim = clock();

  std::cout << "****************************************\n";
  std::cout << "Starting simulation: " << runNo << " (efficiency)\n";
  std::cout << "****************************************\n";

  //Set up variables for simulation
  int totalAvalanches = 0;
  int numAboveThreshold = 0;
  int numNoAvalanche = 0;

  double efficiency = 0.;
  double varience = 0.;
  double efficiencyErr = 0.;

	int numInBunch = 100;//Always do at least 100 avalanches first
  double lowerLimit = 0.;
  double upperLimit = 1.;

  //Begin simulating electron avalanches
  bool runAvalanche = true;
  bool isEfficient = false;
  while(runAvalanche && totalAvalanches < simParams->numAvalanche){
      avalancheE->AvalancheElectron(x0, y0, z0, 0., e0, dx0, dy0, dz0);

      //Electron count - use endpoints to include attached electrons
      int avalancheElectrons = avalancheE->GetNumberOfElectronEndpoints();

      //Increment stats counters
      if(avalancheElectrons == 1){
        numNoAvalanche++;
      }
      if(avalancheElectrons >= electronThreshold){
        numAboveThreshold++;
      }

    }//end of avalanche bunch loop

		numInBunch = 25;//do bunches of 25 after first iteration

    //Efficiency calculations
    double success = numAboveThreshold;
    double total = totalAvalanches - numNoAvalanche;

    //Binomial Stats
    //efficiency = success/total;
    //varience = (efficiency*(1-efficiency)/total;

    //Bayesian Statistics
    efficiency = (success+1)/(total+2);
    varience = ((success+1)*(success+2))/((total+2)*(total+3)) - efficiency*efficiency;
   
    efficiencyErr = std::sqrt(varience);

    // *** Check efficiency ***
    lowerLimit = efficiency - confidenceValue*efficiencyErr;
    upperLimit = efficiency + confidenceValue*efficiencyErr;

    //Efficiency excludes target within confidence
    if(upperLimit < targetEfficiency){
      runAvalanche = false;
      isEfficient = false;
    }
    
    //Efficiency is above target within confidence
    if(lowerLimit >= targetEfficiency){
      runAvalanche = false;
      isEfficient = true;
    }

		//occasionally print values
		if(totalAvalanches%100 == 0){
			std::cout << "Total avalanches: " << totalAvalanches << "\n";
      std::cout << "\tEfficiency: " << efficiency << " +/- " << efficiencyErr << "\n";
		}

  }//end gain convergence loop
  

	//***** Output efficiency value *****//	
	//create output file
  std::string dataFilename = "efficiencyFile.dat";
  std::string dataPath = "../../Data/"+dataFilename;
	std::ofstream dataFile;

	//Write results to file
	dataFile.open(dataPath);
  if(!dataFile.is_open()){
    std::cerr << "Error: Could not open file: " << dataPath << std::endl;
  }

	//write some extra information
	dataFile << "// Finding efficiency for run: " << runNo << "\n";
	dataFile << "// Total avalanches: " << totalAvalanches << " (of " << numAvalanche << ")\n";

  //include convergence criteria
  dataFile << "// Stop condition:\n";
  if(runAvalanche){
    dataFile << "DID NOT CONVERGE\n";
  }
  else{
    if(isEfficient){
      dataFile << "CONVERGED\n";
    }
    else{
      dataFile << "EXCLUDED\n";
    }
  }

  //output efficiency
  dataFile << "// Efficiency:\n" << efficiency << "\n" << efficiencyErr << std::endl;

	dataFile.close();

  //Final timing
  stopSim = clock();
  runTime = (stopSim - startSim)/CLOCKS_PER_SEC;
  std::cout << "****************************************\n";
  std::cout << "Done processing avalanches...(" << runTime << " s)\n";
  std::cout << "****************************************\n";
  std::cout << "Done simulation: " << runNo << " (efficiency)\n";
  std::cout << "****************************************\n";
  std::cout << std::endl;

  return 0;

}
