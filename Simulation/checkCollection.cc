/*
 * checkCollection.cc
 *
 * 
 * Garfield++ simulation of single-electron avalanches in a FIMS geometry.
 * 

 * Input parameters are:
 * <Initial Z position>
 * 
 * Results are written to a file: "collectionFile.dat"
 * 
 * Tanner Polischuk & James Harrison IV
 */

// My includes
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
#include <random>
#include <utility>

using namespace Garfield;

int main(int argc, char * argv[]) {

  const double MICRON = 1e-6;
  const double CM = 1e-2;
  const double MICRONTOCM = 1e-4;

  if(argc != 2){
    std::cerr << "Format: " << argv[0] << " <Initial Z position>" << std::endl;
    return 1;
  }

  double initialZ = std::stod(argv[1])*MICRONTOCM;

  //Random seed
  std::srand(static_cast<unsigned int>(std::time(nullptr)));

  //*************** SETUP ***************//
  //Timing variables
  clock_t startSim, stopSim, runTime;

  //***** Simulation Parameters *****//
  auto simParams = readSimulationParameters();
  if(!simParams){
    return -1;
  }
  int runNo = simParams->runNumber;
  double pitch = simParams->pitch;

  //*************** SIMULATION ***************//

  std::cout << "****************************************\n";
  std::cout << "Building simulation: " << runNo << " (Charge Collection)\n";
  std::cout << "****************************************\n";

  // Define and initialize the gas mixture
  MediumMagboltz* gasFIMS = initializeGas(*simParams);
  if (!gasFIMS) {
    std::cerr << "Error: Failed to initialize gas mixture." << std::endl;
    return -1;
  }

  std::cout << "Loading field map..." << std::endl;
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

  std::cout << "Setting geometry boundaries..." << std::endl;
  // Get region of elmer geometry
  double xmin, ymin, zmin, xmax, ymax, zmax;
  fieldFIMS.GetBoundingBox(xmin, ymin, zmin, xmax, ymax, zmax);

  //Define boundary region for simulation
  double xBoundary[2], yBoundary[2], zBoundary[2];
  zBoundary[0] = zmin;
  zBoundary[1] = zmax;
  //Extend simulation boundary to +/- 2*pitch in x and y
  xBoundary[0] = -2.*pitch;
  xBoundary[1] = 2.*pitch;
  yBoundary[0] = -2.*pitch;
  yBoundary[1] = 2.*pitch;

  //Enable periodicity and set components
  fieldFIMS.EnableMirrorPeriodicityX();
  fieldFIMS.EnableMirrorPeriodicityY();
  fieldFIMS.SetGas(gasFIMS);

  std::cout << "Creating sensor..." << std::endl;
  //Create a sensor
  Sensor* sensorFIMS = new Sensor();
  sensorFIMS->AddComponent(&fieldFIMS);
  sensorFIMS->SetArea(
    xBoundary[0], yBoundary[0], zBoundary[0], 
    xBoundary[1], yBoundary[1], zBoundary[1]
  );

  std::cout << "Enabling avalanche..." << std::endl;
  AvalancheMicroscopic* avalancheE = new AvalancheMicroscopic;
  avalancheE->SetSensor(sensorFIMS);
  avalancheE->EnableAvalancheSizeLimit(5);
  {
    SilenceCerr guard;
    avalancheE->EnablePlotting(nullptr, 10);
  }
  
  // ***** Prepare Avalanche Electron ***** //
  //Set the Initial electron parameters
  double x0 = 0., y0 = 0., z0 = initialZ;
  double t0 = 0.;//ns
  double e0 = 0.1;//eV (Garfield is weird when this is 0.)
  double dx0 = 0., dy0 = 0., dz0 = 0.;//No velocity
 
  //Start timing the sim
  startSim = clock();

  std::cout << "****************************************\n";
  std::cout << "Starting simulation: " << runNo << " (Charge Collection)\n";
  std::cout << "****************************************\n";

  //Begin simulating electron avalanches
  int numSuccess = 0;
  int numTotal = 0;

  int avalancheCheck = (simParams->numAvalanche / 10 > 0) ? (simParams->numAvalanche / 10) : 1;
  int num7 = 0, num5 = 0, num1 = 0;
  int numTrials = 0, numFailure = 0;

  double cellLength = simParams->pitch/std::sqrt(3.);
  
  for(int inAvalanche = 0; inAvalanche < simParams->numAvalanche; inAvalanche++){

    if(inAvalanche % avalancheCheck == 0){
      std::cout << "Avalanche " << inAvalanche << "/" << simParams->numAvalanche << "\n";
    }
    
    // Generate random point in hexagon
    //auto [sampleX, sampleY] = randomXYInHexagon(cellLength);
    //double curX = sampleX, curY = sampleY, curZ = z0;
    double curX = x0, curY = y0, curZ = z0;
    double curTime = t0;
    double curEnergy = e0;
    double curDx = 0., curDy = 0., curDz = 0.;

    double xi, yi, zi, ti, Ei;
    double xf, yf, zf, tf, Ef;
    int stat;


    bool repopulate = true;
    while(repopulate){
      numTrials++;
      avalancheE->AvalancheElectron(curX, curY, curZ, curTime, curEnergy, curDx, curDy, curDz);

      //Electron count - use endpoints to include attached electrons
      int avalancheElectrons = avalancheE->GetNumberOfElectronEndpoints();

      if(avalancheElectrons > 1){
        repopulate = false;
        numSuccess++;
        numTotal++;
        break;
      }

      if(avalancheElectrons != 0){
        avalancheE->GetElectronEndpoint(0, xi, yi, zi, ti, Ei, xf, yf, zf, tf, Ef, stat);
      }
      else{
        std::cerr << "Error: No electrons in avalanche. This should not happen." << std::endl;
        numFailure++;
        stat = -7; //Treat as attached to trigger repopulation
      }
      
      switch(stat){

        case -7: {// Electron attached to gas molecule - Restart with initial electron
          //WARNING - This may cause an infinite loop. Consider max attempts if becomes an issue
          //auto [sampleX, sampleY] = randomXYInHexagon(cellLength);
          //curX = sampleX, curY = sampleY, curZ = z0;
          curX = x0, curY = y0, curZ = z0;
          curTime = t0;
          curEnergy = e0;
          curDx = 0., curDy = 0., curDz = 0.;
          num7++;
          break;  
        }       

        case -5: {// Electron leave drift medium (Hits Grid/Pad/Dielectric)
          repopulate = false;
          numTotal++;
          // Check if below grid - TODO: Courrently just checking final z
          if(zf < -1.*simParams->gridThickness){
            numSuccess++;
          }
          num5++;
          break;
        }

        case -1: {// Electron leaves drift area (Simulation Volume) - Shift it back
          //Determine which boundary was hit and shift by pitch to opposite side
          //Example: If leaves volume at x=+2pitch, move to x=-pitch
          //Allows direction vector to also just be translated
          curX = std::abs(xf) >= .99*2.*pitch ? -1.*std::copysign(pitch, xf) : xf;
          curY = std::abs(yf) >= .99*2.*pitch ? -1.*std::copysign(pitch, yf) : yf;
          curZ = std::abs(zf) <= 1.01*simParams->gridThickness/2. ? std::copysign(simParams->gridThickness, zf) : zf;

          curTime = tf;
          curEnergy = Ef;

          //Get 2nd-last and last points along drift
          double xPrev, yPrev, zPrev, tPrev, xFinal, yFinal, zFinal, tFinal;
          int nPoints = avalancheE->GetNumberOfElectronDriftLinePoints(0);
          avalancheE->GetElectronDriftLinePoint(xFinal, yFinal, zFinal, tFinal, nPoints-1, 0);
          avalancheE->GetElectronDriftLinePoint(xPrev, yPrev, zPrev, tPrev, nPoints-2, 0);

          //Get normalized direction vector
          double dx = xFinal - xPrev;
          double dy = yFinal - yPrev;
          double dz = zFinal - zPrev;
          double vMag = std::sqrt(dx*dx + dy*dy + dz*dz);
          curDx = dx/vMag;
          curDy = dy/vMag;
          curDz = dz/vMag;
          
          num1++;
          break;
        }

        default:
          std::cerr << "Error: Unexpected electron endpoint status: " << stat << std::endl;
          return -1;
      }
    }
  }

  delete avalancheE;
  delete sensorFIMS;
  delete gasFIMS;

  std::cout << "Electron endpoint status counts:\n";
  std::cout << "\tElectrons populated: " << numTrials << "\n";
  std::cout << "\tElectron attached to gas molecule (-7): " << num7 << "\n";
  std::cout << "\tElectron leave drift medium (Hits Grid/Pad/Dielectric) (-5): " << num5 << "\n";
  std::cout << "\tElectron leaves drift area (Simulation Volume) (-1): " << num1 << "\n";
  std::cout << "\tNumber of failures (no electrons in avalanche) (These are included in -7): " << numFailure << "\n";

  //Charge Collection Efficiency calculations - Bayesian Statistics
  double success = 1.*numSuccess;
  double total = 1.*numTotal;
  double efficiency = (success+1)/(total+2);
  double varience = ((success+1)*(success+2))/((total+2)*(total+3)) - efficiency*efficiency;
  double efficiencyErr = std::sqrt(varience);
  
	//***** Output efficiency value *****//	
	//create output file
  std::string dataFilename = "collectionFile.dat";
  std::string dataPath = "../../Data/"+dataFilename;
	std::ofstream dataFile;

	//Write results to file
	dataFile.open(dataPath);
  if(!dataFile.is_open()){
    std::cerr << "Error: Could not open file: " << dataPath << std::endl;
  }

	//write some extra information
	dataFile << "// Finding collection efficiency for run: " << runNo << "\n";
  dataFile << "// Field Ratio: " << simParams->fieldRatio << "\n";
	dataFile << "// Total avalanches: " << simParams->numAvalanche  << "\n";

  //include convergence criteria
  dataFile << "// Stop condition:\n";
  dataFile << "TRIALS COMPLETE\n";
  

  //output efficiency
  dataFile << "// Collection:\n" << efficiency << "\n" << efficiencyErr << std::endl;

	dataFile.close();

  //Final timing
  stopSim = clock();
  runTime = (stopSim - startSim)/CLOCKS_PER_SEC;
  std::cout << "****************************************\n";
  std::cout << "Done processing avalanches...(" << runTime << " s)\n";
  std::cout << "****************************************\n";
  std::cout << "Done simulation: " << runNo << " (Charge Collection)\n";
  std::cout << "****************************************\n";
  std::cout << std::endl;

  return 0;

}
