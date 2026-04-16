/*
 * checkCharge.cc
 *
 * 
 * Garfield++ simulation of single-electron avalanches in a FIMS geometry.
 * 

 * Input parameters are:
 * <Initial Z position>
 * 
 * Results are written to a file: "chargeFile.dat"
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
#include "Garfield/ViewDrift.hh"

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

  if(argc != 2){
    std::cerr << "Format: " << argv[0] << " <Initial Z position>" << std::endl;
    return 1;
  }

  double initialZ = std::atof(argv[1]);

  //Random seed
  std::srand(static_cast<unsigned int>(std::time(nullptr)));

  const double MICRON = 1e-6;
  const double CM = 1e-2;
  const double MICRONTOCM = 1e-4;
  
  //*************** SETUP ***************//
  //Timing variables
  clock_t startSim, stopSim, runTime;

  //***** Simulation Parameters *****//
  auto simParams = readSimulationParameters();
  if(!simParams){
    return -1;
  }
  int runNo = simParams->runNumber;

  //*************** SIMULATION ***************//

  std::cout << "****************************************\n";
  std::cout << "Building simulation: " << runNo << " (charge)\n";
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
  ViewDrift* viewElectronDrift = new ViewDrift();
  avalancheE->SetSensor(sensorFIMS);
  avalancheE->EnableAvalancheSizeLimit(5);

  viewElectronDrift->SetArea(
    xBoundary[0], yBoundary[0], zBoundary[0], 
    xBoundary[1], yBoundary[1], zBoundary[1]
  );
  avalancheE->EnablePlotting(viewElectronDrift, 250);
      
  // ***** Prepare Avalanche Electron ***** //
  //Set the Initial electron parameters
  double x0 = 0., y0 = 0., z0 = initialZ;
  double t0 = 0.;//ns
  double e0 = 0.1;//eV (Garfield is weird when this is 0.)
  double dx0 = 0., dy0 = 0., dz0 = 0.;//No velocity
  
  //Start timing the sim
  startSim = clock();

  std::cout << "****************************************\n";
  std::cout << "Starting simulation: " << runNo << " (efficiency)\n";
  std::cout << "****************************************\n";

  //Begin simulating electron avalanches
  int numSuccess = 0;
  int numTotal = 0;

  for(int inAvalanche = 0; inAvalanche < simParams->numAvalanche; inAvalanche++){

    double curX = x0, curY = y0, curZ = z0;
    double curTime = t0;
    double curEnergy = e0;
    double curDx = 0., curDy = 0., curDz = 0.;

    double xi, yi, zi, ti, Ei;
    double xf, yf, zf, tf, Ef;
    int stat;

    bool repopulate = true;
    while(repopulate){
      avalancheE->AvalancheElectron(curX, curY, curZ, curTime, curEnergy, curDx, curDy, curDz);

      //Electron count - use endpoints to include attached electrons
      int avalancheElectrons = avalancheE->GetNumberOfElectronEndpoints();

      if(avalancheElectrons > 1){
        repopulate = false;
        numSuccess++;
        numTotal++;
        break;
      }

      avalancheE->GetElectronEndpoint(0, xi, yi, zi, ti, Ei, xf, yf, zf, tf, Ef, stat);

      switch(stat){
        // -7: 
        // -5: Electron leaves drift medium (Hits: Grid, Pad, Dielectric)
        // -1: Electron leave drift area (Simulation Volume)

        case -7: // Electron attached to gas molecule - Restart with initial electron
          //WARNING - This may cause an infinite loop. Consider max attempts if becomes an issue
          curX = x0, curY = y0, curZ = z0;
          curTime = t0;
          curEnergy = e0;
          curDx = 0., curDy = 0., curDz = 0.;
          break;         

        case -5: // Electron leave drift medium (Hits Grid/Pad/Dielectric)
          repopulate = false;
          numTotal++;
          // Check if below grid
          if(zf < -1.*simParams->gridThickness){
            numSuccess++;
          }
          break;

        case -1: {// Electron leaves drift area (Simulation Volume) - Reflect it back
          //Determine which boundary was hit and shift slightly back into volume
          double boundaryOffset = simParams->pitch - 1.*MICRONTOCM;
          if(std::abs(xf) >= simParams->pitch){
            curX = std::copysign(boundaryOffset, xf);
          }
          else{
            curX = xf;
          }

          if(std::abs(yf) >= simParams->pitch){
            curY = std::copysign(boundaryOffset, yf);
          }
          else{
            curY = yf;
          }
          curZ = zf;

          curTime = tf;
          curEnergy = Ef;

          //Find direction via drift lines
          int numElectronDrift = viewElectronDrift->GetNumberOfDriftLines();
          bool isElectron;
          std::vector<std::array<float, 3> > electronDriftLines;
          viewElectronDrift->GetDriftLine(0, electronDriftLines, isElectron);
          int numDriftLines = electronDriftLines.size();

          
          // Get 2nd-last and last points
          double xPrev, yPrev, zPrev, xFinal, yFinal, zFinal;
          xPrev = electronDriftLines[numDriftLines-2][0];
          yPrev = electronDriftLines[numDriftLines-2][1];
          zPrev = electronDriftLines[numDriftLines-2][2];

          xFinal = electronDriftLines[numDriftLines-1][0];
          yFinal = electronDriftLines[numDriftLines-1][1];
          zFinal = electronDriftLines[numDriftLines-1][2];

          double dx = xFinal - xPrev;
          double dy = yFinal - yPrev;
          double dz = zFinal - zPrev;
          double vMag = std::sqrt(dx*dx + dy*dy + dz*dz);
          dx /= vMag;
          dy /= vMag;
          dz /= vMag;

          if(std::abs(xf) >= simParams->pitch){
            curDx = -dx;
          }
          else{
            curDx = dx;
          }

          if(std::abs(yf) >= simParams->pitch){
            curDy = -dy;
          }
          else{
            curDy = dy;
          }
          curDz = dz;

          viewElectronDrift->Clear();
          break;
        }

        default:
          std::cerr << "Error: Unexpected electron endpoint status: " << stat << std::endl;
          return -1;
      }
    }
  }

  //Charge Efficiency calculations - Bayesian Statistics
  double success = 1.*numSuccess;
  double total = 1.*numTotal;
  double efficiency = (success+1)/(total+2);
  double varience = ((success+1)*(success+2))/((total+2)*(total+3)) - efficiency*efficiency;
  double efficiencyErr = std::sqrt(varience);
  
	//***** Output efficiency value *****//	
	//create output file
  std::string dataFilename = "chargeFile.dat";
  std::string dataPath = "../../Data/"+dataFilename;
	std::ofstream dataFile;

	//Write results to file
	dataFile.open(dataPath);
  if(!dataFile.is_open()){
    std::cerr << "Error: Could not open file: " << dataPath << std::endl;
  }

	//write some extra information
	dataFile << "// Finding charge efficiency for run: " << runNo << "\n";
  dataFile << "// Field Ratio: " << simParams->fieldRatio << "\n";
	dataFile << "// Total avalanches: " << simParams->numAvalanche  << "\n";

  //include convergence criteria
  dataFile << "// Stop condition:\n";
  dataFile << "All trials complete\n";
  

  //output efficiency
  dataFile << "// Charge:\n" << efficiency << "\n" << efficiencyErr << std::endl;

	dataFile.close();

  //Final timing
  stopSim = clock();
  runTime = (stopSim - startSim)/CLOCKS_PER_SEC;
  std::cout << "****************************************\n";
  std::cout << "Done processing avalanches...(" << runTime << " s)\n";
  std::cout << "****************************************\n";
  std::cout << "Done simulation: " << runNo << " (charge)\n";
  std::cout << "****************************************\n";
  std::cout << std::endl;

  return 0;

}
