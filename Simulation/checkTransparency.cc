/*
 * fieldlines.cc
 * 
 * Garfield++ mapping of electric field lines to determine the electric field transparency
 *    Requires an input electric field solved by elmer and geometry from gmsh.
 *    Reads simulation parameters from runControl.
 * 
 * Tanner Polischuk & James E. Harrison IV
 */

//My includes
#include "SilenceConsole.h"
#include "myFunctions.h"

//Garfield includes
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

using namespace Garfield;

int main(int argc, char * argv[]) {

  std::srand(static_cast<unsigned int>(std::time(nullptr)));

  const double MICRON = 1e-6;
  const double CM = 1e-2;
  const double MICRONTOCM = 1e-4;
  bool DEBUG = false;
  double fieldLineX, fieldLineY, fieldLineZ;
  
  //*************** SETUP ***************//
  //Timing variables
  clock_t startSim, stopSim, runTime;

  //***** Simulation Parameters *****//
  //Read in simulation parameters from stdin as JSON
  std::cout << "****************************************\n";
  std::cout << "Setting up field line simulation.\n";
  std::cout << "****************************************\n";

  // Read JSON parameters from stdin
  auto simParams = readSimulationParameters();
  if(!simParams){
    return -1;
  }
  int runNo = simParams->runNumber;

  std::cout << "****************************************\n";
  std::cout << "Building field line simulation: " << runNo << "\n";
  std::cout << "****************************************\n";

  
  //*************** SIMULATION ***************//
  std::cout << "****************************************\n";
  std::cout << "Creating field line simulation: " << "\n";
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

  //Create a sensor
  Sensor* sensorFIMS = new Sensor();
  sensorFIMS->AddComponent(&fieldFIMS);
  sensorFIMS->SetArea(xBoundary[0], yBoundary[0], zBoundary[0], xBoundary[1], yBoundary[1], zBoundary[1]);
  sensorFIMS->AddElectrode(&fieldFIMS, "wtlel");

  // ***** Find field transparency ***** //
  std::cout << "****************************************\n";
  std::cout << "Calculating field Lines.\n";
  std::cout << "****************************************\n";

  DriftLineRKF driftLines(sensorFIMS);
  driftLines.SetMaximumStepSize(MICRONTOCM);

  std::vector<double> xStart;
  std::vector<double> yStart;
  double rangeScale = 0.99;
  double fieldCutoff = 0.2;
  double xRange = (xBoundary[1] - xBoundary[0])*rangeScale;
  double yRange = (yBoundary[1] - yBoundary[0])*rangeScale;
  double xWidth = rangeScale*simParams->pitch*sqrt(3.)/3.;
  double yWidth = rangeScale*simParams->pitch/2.;

  // ***** Generate field line start points ***** //
  
  /*
  //Lines generated radially from the center to corner of unit cell
  //The x-direction is the long axis of the geometry. 
  for(int i = 0; i < simParams->numFieldLine; i++){
    xStart.push_back(xWidth*i/(simParams->numFieldLine-1));
    yStart.push_back(0.);
  }
  */


  //Rejection sampled points near the edge of the unit cell
  double sampleWidth = std::sqrt(0.85); // Reject the inner portion of the unit cell
  double safetyWidth = std::sqrt(0.99); // Reject points very close to the edges of the unit cell
  double halfPitch = simParams->pitch/2.;
  double cellLength = simParams->pitch/std::sqrt(3.);
  while(xStart.size() < simParams->numFieldLine){

    // Generate random point in rectangle defined by cellLength (x) and halfPitch (y)
    double sampleX =  ((double)std::rand()/RAND_MAX)*cellLength;
    double sampleY =  ((double)std::rand()/RAND_MAX)*halfPitch;

    //Determine if point is in unit cell - Skip if not
    double unitY = (simParams->pitch/cellLength) * (cellLength-sampleX);
    if(sampleY > unitY){
      continue;
    }

    //Determine if point is near edge of cell - Skip if not
    double checkY = sampleWidth*halfPitch;
    double edgeY = (simParams->pitch/cellLength) * (sampleWidth*cellLength-sampleX);
    if((sampleY < checkY) && (sampleY < edgeY)){
      continue;
    }

    //Ensure point is not too close to edge of cell
    double safetyY = safetyWidth*halfPitch;
    double safetyEdgeY = (simParams->pitch/cellLength) * (safetyWidth*cellLength-sampleX);
    if((sampleY < safetyY) && (sampleY < safetyEdgeY)){
      xStart.push_back(sampleX);
      yStart.push_back(sampleY);
      //std::cout << sampleX << ", " << sampleY << ", ";
    }
  }


  // ***** Calculate field Lines ***** //
  std::vector<std::array<float, 3> > fieldLines;
  int totalFieldLines = xStart.size();
  int numAtPad = 0;
  int prevDriftLine = 0;

  double transparency = 0.;
  double variance = 0.;
  double transparencyErr = 0.;

  std::cout << "Computing field lines" << std::endl;
  for(int inFieldLine = 0; inFieldLine < totalFieldLines; inFieldLine++){

    //Create field line
    driftLines.FieldLine(xStart[inFieldLine], yStart[inFieldLine], zmax*rangeScale, fieldLines);
    
    //Find if termination point is at pad
    int lineEnd = fieldLines.size() - 1;
   
    fieldLineX = fieldLines[lineEnd][0];
    fieldLineY = fieldLines[lineEnd][1];
    fieldLineZ = fieldLines[lineEnd][2];

    // Check if final z is within half the SiO2 thickness of the pad
    // Line may terminate on any pad without issue
    if(fieldLineZ < simParams->thicknessSiO2/2. - simParams->gridStandoff){
        numAtPad++;
    }

    //Print a progress update every 10%
    int driftLineProgress = (100*(inFieldLine+1))/totalFieldLines;
    if(   (driftLineProgress % 10 == 0)
      &&  (driftLineProgress != prevDriftLine)){
      std::cout << "\tDriftline Progress: " << driftLineProgress << " %" << std::endl;
      prevDriftLine = driftLineProgress;
    }

  }//End field line loop

  std::cout << "Done " << totalFieldLines << " field lines; Determining transparency." << "\n";

  //Determine transparency - Binomial Statistics
  //transparency = (1.*numAtPad) / (1.*simParams->numFieldLine);
  //transparencyErr = sqrt(transparency*(1-transparency)/simParams->numFieldLine);

  //Determine transparency - Bayesian statistics
  double success = 1.*numAtPad;
  double total = 1.*simParams->numFieldLine;

  transparency = (success + 1.) / (total + 2.);
  variance = ((success+1.)*(success+2.))/((total+2.)*(total+3.)) - transparency*transparency;
  transparencyErr = std::sqrt(variance);


  std::cout << "Transparency is " << transparency <<  "." << std::endl;

  //***** Output transparency value *****//	
	//create output file
  std::string dataFilename = "fieldTransparency.dat";
  std::string dataPath = "../../Data/"+dataFilename;
  std::ofstream dataFile;

  dataFile.open(dataPath);
  if(!dataFile.is_open()){
    std::cerr << "Error: Could not open file: " << dataPath << std::endl;
  }

  //write some extra information
	dataFile << "// Finding transparency for run: " << runNo << "\n";
	dataFile << "// Field lines at pad: " << numAtPad << " (of " << simParams->numFieldLine << ")\n";

  //***** Output transparency value *****//
  dataFile << "// Transparency:\n" << transparency << "\n" << transparencyErr << std::endl;

  dataFile.close();

  std::cout << "****************************************\n";
  std::cout << "Finished field line simulation " << "\n";
  std::cout << "****************************************\n";
  std::cout << std::endl;

  return 0;

}
