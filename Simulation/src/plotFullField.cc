/*
 * plotFullField.cc
 * 
 * Garfield++ mapping of electric field lines for visualization
 *    Requires an input electric field solved by elmer and geometry from gmsh.
 *    Reads simulation parameters from runControl.
 * 
 * Tanner Polischuk & James E. Harrison IV
 */

// My includes
#include "myFunctions.hh"

// Garfield includes
#include "Garfield/ComponentElmer.hh"
#include "Garfield/AvalancheMicroscopic.hh"
#include "Garfield/MediumMagboltz.hh"
#include "Garfield/AvalancheMC.hh"
#include "Garfield/Sensor.hh"
#include "Garfield/DriftLineRKF.hh"

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
#include <random>

using namespace Garfield;

int main(int argc, char * argv[]) {
  if(argc != 2){
      std::cerr << "Format: " << argv[0] << " <GeometryMode>" << std::endl;
      return -1;
  }
  
  std::cout << "****************************************\n";
  std::cout << "Building field line simulation: " << "\n";
  std::cout << "****************************************\n";
  
  // Determine the geometry layout of the simulation
  std::string geoModeString = argv[1];
  GeometryMode geometryMode = stringToGeometryMode(argv[1]);
  if(geometryMode == GeometryMode::Unknown){
      std::cerr << "Error: Invalid GeometryMode: " << argv[1] << std::endl;
      return -1;
  }
  
  // Read in simulation parameters
  auto simParams = readSimulationParameters();
  if(!simParams){
      return -1;
  }
  
  // Miscellaneous constants
  std::srand(static_cast<unsigned int>(std::time(nullptr)));
  const double MICRON = 1e-6;
  const double CM = 1e-2;
  const double MICRONTOCM = 1e-4;
  bool DEBUG = false;
  double fieldLineX, fieldLineY, fieldLineZ;
  clock_t startSim, stopSim, runTime;
  
  std::cout << "****************************************\n";
  std::cout << "Setting up field line simulation.\n";
  std::cout << "****************************************\n";
  
  // Define the gas mixture
  MediumMagboltz* gasFIMS = initializeGas(*simParams); 
  
  // Import elmer-generated field map
  std::string geometryPath = "../Geometry/";
  std::string elmerResultsPath = geometryPath+"elmerResults/";
  ComponentElmer fieldFIMS(
    elmerResultsPath+"mesh.header",
    elmerResultsPath+"mesh.elements",
    elmerResultsPath+"mesh.nodes", 
    geometryPath+"dielectrics.dat",
    elmerResultsPath+geoModeString+".result", 
    "mum"
  );

  // Get region of elmer geometry
  double xmin, ymin, zmin, xmax, ymax, zmax;
  fieldFIMS.GetBoundingBox(xmin, ymin, zmin, xmax, ymax, zmax);

  // Define boundary region for simulation
  double xBoundary[2], yBoundary[2], zBoundary[2];

  // Simple criteria for if 1/4 geometry or full
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

  // Enable periodicity and set components
  fieldFIMS.EnableMirrorPeriodicityX();
  fieldFIMS.EnableMirrorPeriodicityY();
  fieldFIMS.SetGas(gasFIMS);

  // Create a sensor
  Sensor* sensorFIMS = new Sensor();
  sensorFIMS->AddComponent(&fieldFIMS);
  sensorFIMS->SetArea(xBoundary[0], yBoundary[0], zBoundary[0], xBoundary[1], yBoundary[1], zBoundary[1]);
  sensorFIMS->AddElectrode(&fieldFIMS, "wtlel");

  // ***** Calculate Field Lines ***** //
  std::cout << "****************************************\n";
  std::cout << "Calculating field Lines.\n";
  std::cout << "****************************************\n";

  DriftLineRKF driftLines(sensorFIMS);
  driftLines.SetMaximumStepSize(MICRONTOCM);

  std::vector<double> xStart;
  std::vector<double> yStart;
  double rangeScale = 0.99;

  // ***** Generate field line start points ***** //
  double safetyWidth = std::sqrt(0.98);
  double cellLength = safetyWidth * simParams->pitch/std::sqrt(3.0);
    
  while(xStart.size() < simParams->numFieldLine){
      auto [sampleX, sampleY] = randomXYInHexagon(cellLength);
      xStart.push_back(sampleX);
      yStart.push_back(sampleY);
  }
  
  // ***** Calculate field Lines ***** //
  std::vector<std::array<float, 3> > fieldLines;
  int totalFieldLines = xStart.size();
  int prevDriftLine = 0;

  std::cout << "Computing field lines" << std::endl;
  // Create and open field line data file
  
  std::string fieldFileName = "sim"+std::to_string(simParams->runNumber)+"fullFieldLines.dat";
  std::string fieldFilePath = "../../Data/"+fieldFileName;
  std::ofstream fieldFile;
  fieldFile.open(fieldFilePath, std::ios::out);
  
  if(!fieldFile.is_open()){
    std::cerr << "Error: Could not open file: " << fieldFilePath << std::endl;
  }

  for(int inFieldLine = 0; inFieldLine < totalFieldLines; inFieldLine++){

    // Create field line
    driftLines.FieldLine(xStart[inFieldLine], yStart[inFieldLine], zmax*rangeScale, fieldLines);
    
    // Print Field Line points to file
    for(int inLine = 0; inLine < fieldLines.size(); inLine++){
        fieldLineX = fieldLines[inLine][0];
        fieldLineY = fieldLines[inLine][1];
        fieldLineZ = fieldLines[inLine][2];
        fieldFile << fieldLineX << ", " << fieldLineY << ", " << fieldLineZ << "\n";
    }

    //Print a progress update every 10%
    int driftLineProgress = (100*(inFieldLine+1))/totalFieldLines;
    if(   (driftLineProgress % 10 == 0)
      &&  (driftLineProgress != prevDriftLine)){
      std::cout << "\tDriftline Progress: " << driftLineProgress << " %" << std::endl;
      prevDriftLine = driftLineProgress;
    }

  }//End field line loop
  fieldFile.close();

  std::cout << "****************************************\n";
  std::cout << "Finished field line simulation " << "\n";
  std::cout << "****************************************\n";
  std::cout << std::endl;

  return 0;

}
