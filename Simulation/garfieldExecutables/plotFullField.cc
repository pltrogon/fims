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
#include "SilenceConsole.h"

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

  std::srand(static_cast<unsigned int>(std::time(nullptr)));

  const double MICRON = 1e-6;
  const double CM = 1e-2;
  const double MICRONTOCM = 1e-4;
  bool DEBUG = false;
  double fieldLineX, fieldLineY, fieldLineZ;
  
  //*************** SETUP ***************//
  // Timing variables
  clock_t startSim, stopSim, runTime;
  
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
  std::cout << "Building field line simulation: " << "\n";
  std::cout << "****************************************\n";
  
  //***** Simulation Parameters *****//
  // Read in simulation parameters from runControl
  int numInputs;
  double  padLength, pitch;
  double gridStandoff, gridThickness, holeRadius;
  double cathodeHeight, thicknessSiO2, pillarRadius;
  double fieldRatio;
  int numFieldLine;
  double gasCompAr, gasCompCO2, gasCompCF4, gasCompIsobutane;
  double gasPenning;

  std::ifstream paramFile;
  std::string runControlFile = "../runControl";
  paramFile.open(runControlFile);

  if(!paramFile.is_open()){
    std::cerr << "Error: Could not open control file '" << runControlFile << "'." << std::endl;
    return -1;
  }

  std::cout << "****************************************\n";
  std::cout << "Setting up field line simulation.\n";
  std::cout << "****************************************\n";
  
  std::string curLine;
  std::map<std::string, std::string> readParam;

  // Read the file contents to a map
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

  // Parse the values from the map
  numInputs = std::stoi(readParam["numInputs"]);
  if(numKeys != numInputs){
    std::cerr << "Error: Invalid simulation parameters in 'runControl'." << std::endl;
    return -1;
  }

  // Geometry parameters
  // Garfield's operational scale is cm. runControl is defined in microns
  padLength = std::stod(readParam["padLength"])*MICRONTOCM;
  pitch = std::stod(readParam["pitch"])*MICRONTOCM;

  gridStandoff = std::stod(readParam["gridStandoff"])*MICRONTOCM;
  gridThickness = std::stod(readParam["gridThickness"])*MICRONTOCM;
  holeRadius = std::stod(readParam["holeRadius"])*MICRONTOCM;

  cathodeHeight = std::stod(readParam["cathodeHeight"])*MICRONTOCM;
  thicknessSiO2 = std::stod(readParam["thicknessSiO2"])*MICRONTOCM;
  pillarRadius = std::stod(readParam["pillarRadius"])*MICRONTOCM;

  // Field parameters
  fieldRatio = std::stod(readParam["fieldRatio"]);
  numFieldLine = std::stoi(readParam["numFieldLine"]);

  // Simulation Parameters
 // Gasses defined as percentages
  gasCompAr = std::stod(readParam["gasCompAr"])*100.;
  gasCompCO2 = std::stod(readParam["gasCompCO2"])*100.;
  gasCompCF4 = std::stod(readParam["gasCompCF4"])*100.;
  gasCompIsobutane = std::stod(readParam["gasCompIsobutane"])*100.;

  gasPenning = std::stod(readParam["gasPenning"]);

  
  //*************** SIMULATION ***************//
  std::cout << "****************************************\n";
  std::cout << "Creating field line simulation: " << "\n";
  std::cout << "****************************************\n";
  
  // Define the gas mixture
  MediumMagboltz* gasFIMS = new MediumMagboltz();

  // Set gas parameters
  gasFIMS->SetComposition(
    "ar", gasCompAr, 
    "co2", gasCompCO2,
    "cf4", gasCompCF4,
    "iC4H10", gasCompIsobutane
  );

  {
	SilenceCerr guard; 
	gasFIMS->EnablePenningTransfer(gasPenning, 0.0, "ar");
  }
  
  gasFIMS->SetTemperature(293.15); // Room temperature
  gasFIMS->SetPressure(760.);     // Atmospheric pressure
  gasFIMS->SetMaxElectronEnergy(200);
  gasFIMS->Initialise(true);
  
  // Load the ion mobilities.
  // Load ion mobilities. 
  const std::string path = std::getenv("GARFIELD_INSTALL");
  const std::string posIonPath = path + "/share/Garfield/Data/IonMobility_Ar+_Ar.txt";
  //const std::string negIonPath = path + "/share/Garfield/Data/IonMobility_CO2+_CO2.txt";
  const std::string negIonPath = path + "/share/Garfield/Data/IonMobility_CF4+_CF4.txt";
  gasFIMS->LoadIonMobility(posIonPath);
  gasFIMS->LoadNegativeIonMobility(negIonPath);//TODO - Is this correct for negative ion drift? 
  
  // Import elmer-generated field map
  std::string geometryPath = "../Geometry/";
  std::string elmerResultsPath = geometryPath+"elmerResults/";
  ComponentElmer fieldFIMS(
    elmerResultsPath+"mesh.header",
    elmerResultsPath+"mesh.elements",
    elmerResultsPath+"mesh.nodes", 
    geometryPath+"dielectrics.dat",
    elmerResultsPath+"FIMSSurrounding.result", 
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
  const double sqrt3 = std::sqrt(3.0);
  
  double halfPitch = pitch/2.;
  double cellLength = pitch/sqrt3;
    
  while(xStart.size() < numFieldLine){
      static std::mt19937 rng(std::random_device{}());
      std::uniform_real_distribution<double> dist(-1.0, 1.0);

      // Uniform sample in box
      double sampleX = dist(rng)*cellLength*safetyWidth;
      double sampleY = dist(rng)*halfPitch*safetyWidth;
      
      // Check if in hexagon (use symmetry of Q1)
      double absX = std::fabs(sampleX);
      double absY = std::fabs(sampleY);
      if(absX <= cellLength - absY/sqrt3){
          xStart.push_back(sampleX);
          yStart.push_back(sampleY);
      }
  }
  
  // ***** Calculate field Lines ***** //
  std::vector<std::array<float, 3> > fieldLines;
  int totalFieldLines = xStart.size();
  int prevDriftLine = 0;

  std::cout << "Computing field lines" << std::endl;
  // Create and open field line data file
  
  std::string fieldFileName = "sim"+std::to_string(runNo)+"fullFieldLines.dat";
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
