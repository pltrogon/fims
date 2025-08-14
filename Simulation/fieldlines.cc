/*
 * fieldlines.cc
 * 
 * Garfield++ mapping of electric field lines to determine the electric field transparency
 *    Requires an input electric field solved by elmer and geometry from gmsh.
 *    Reads simulation parameters from runControl.
 * 
 * Tanner Polischuk & James Harrison IV
 */

//Garfield includes
#include "Garfield/ComponentElmer.hh"
#include "Garfield/AvalancheMicroscopic.hh"
#include "Garfield/MediumMagboltz.hh"
#include "Garfield/AvalancheMC.hh"
#include "Garfield/Sensor.hh"
#include "Garfield/DriftLineRKF.hh"
#include "Garfield/ViewDrift.hh"

//ROOT includes
#include "TApplication.h"

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
  int fieldLineID;
  double fieldLineX, fieldLineY, fieldLineZ;
  int gridFieldLineLocation;
  double gridLineX, gridLineY, gridLineZ;
  double fieldTransparency;

  
  //*************** SETUP ***************//
  //Timing variables
  clock_t startSim, stopSim, runTime;

  TApplication app("app", &argc, argv);

  //***** Git Hash *****//
  TString gitVersion = "UNKNOWN VERSION";

  const char * getGitCommand = "git describe --tags --always --dirty 2>/dev/null";

  FILE * pipe = popen(getGitCommand, "r");
  if(!pipe){
    std::cerr << "Error: Could not open pipe to 'getGitCommand'." << std::endl;
  }
  else{ 
    char gitBuffer[128];
    std::string gitOutput = "";
    while(fgets(gitBuffer, sizeof(gitBuffer), pipe) != NULL){
      gitOutput += gitBuffer;
    }
    int gitStatus = pclose(pipe);

    if(!gitOutput.empty() && gitOutput.back() == '\n'){
      gitOutput.pop_back();
    }

    if(gitStatus == 0){
      gitVersion = gitOutput;
    }
    else{
      std::cerr << "Error: 'getGitCommand' failed with status " << gitStatus << std::endl;
    }
  }

  std::cout << "****************************************\n";
  std::cout << "Building simulation: " << "\n";
  std::cout << "****************************************\n";

  //***** Data File *****//
  std::string dataFilename = "fieldTransparency.txt";
  std::string dataPath = "../../Data/"+dataFilename;
  std::ofstream dataFile;
  dataFile.open(dataPath);
  
  
  if(!dataFile.is_open()){
    std::cerr << "Error creating or opening file '" << dataFilename << "'." << std::endl;
    return -1;
  }

  //***** Simulation Parameters *****//
  //Read in simulation parameters from runControl

  double  padLength, pitch;
  double gridStandoff, gridThickness, holeRadius;
  double cathodeHeight, thicknessSiO2;
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
  if(numKeys != 14){//Number of user-defined simulation parameters in runControl to search for.
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
  std::cout << "Creating simulation: " << "\n";
  std::cout << "****************************************\n";
  
  // Define the gas mixture
  MediumMagboltz* gasFIMS = new MediumMagboltz();

  //Set parameters
  gasFIMS->SetComposition("ar", gasCompAr, "co2", gasCompCO2);
  gasFIMS->SetTemperature(293.15); // Room temperature
  gasFIMS->SetPressure(760.);     // Atmospheric pressure
  gasFIMS->SetMaxElectronEnergy(200);
  gasFIMS->Initialise(true);
  // Load the penning transfer and ion mobilities.
  gasFIMS->EnablePenningTransfer(0.51, .0, "ar");

  const std::string path = std::getenv("GARFIELD_INSTALL");
  gasFIMS->LoadIonMobility(path + "/share/Garfield/Data/IonMobility_Ar+_Ar.txt");
  gasFIMS->LoadNegativeIonMobility(path + "/share/Garfield/Data/IonMobility_CO2+_CO2.txt");//TODO - Is this correct for negative ion
  
  // Import elmer-generated field map
  std::string geometryPath = "../Geometry/";
  std::string elmerResultsPath = geometryPath+"elmerResults/";
  ComponentElmer fieldFIMS(elmerResultsPath+"mesh.header",
                           elmerResultsPath+"mesh.elements",
                           elmerResultsPath+"mesh.nodes", 
                           geometryPath+"dielectrics.dat",
                           elmerResultsPath+"FIMS.result", 
                           "mum");

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

  // Import the weighting field for the readout electrode.
  fieldFIMS.SetWeightingField(elmerResultsPath+"FIMSWeighting.result", "wtlel");

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
  driftLines.SetMaximumStepSize(MICRONTOCM/10);

  std::vector<double> xStart;
  std::vector<double> yStart;

  double rangeScale = 0.99;
  double xRange = (xBoundary[1] - xBoundary[0])*rangeScale;
  double yRange = (yBoundary[1] - yBoundary[0])*rangeScale;

  //Lines generated radially from the center to edge of geometry
  //    The x-direction is the long axis of the geometry. 
  //    This extends past the vertex of the hex unit cell
  for(int i = 0; i < numFieldLine; i++){
    xStart.push_back((2/3)*xBoundary[1]*i/(numFieldLine-1));
    yStart.push_back(0.);
  }
  
  /*
  //Lines populated at corner - spread with uniform random numbers
  //TODO - now with hex geomety, do some math to find that corner
  double spreadScale = 0.99995; //Any smaller and goes out of bounds
  for(int i = 0; i < numFieldLine; i++){
    //Get random numbers between 0 and 1
    double randX = 1.0*rand()/RAND_MAX;
    double randY = 1.0*rand()/RAND_MAX;
    xStart.push_back(xRange/2. + (randX - 0.5)*(1-spreadScale));
    yStart.push_back(yRange/2. + (randY - 0.5)*(1-spreadScale));
  }
  */

  // ***** Calculate field Lines ***** //
  std::vector<std::array<float, 3> > fieldLines;
  int totalFieldLines = xStart.size();
  int numAtPad = 0;
  std::cout << "Computing " << totalFieldLines << " field lines." << std::endl;
  int prevDriftLine = 0;
  for(int inFieldLine = 0; inFieldLine < totalFieldLines; inFieldLine++){
    fieldLineID = inFieldLine;
    //Calculate from top of volume
    driftLines.FieldLine(xStart[inFieldLine], yStart[inFieldLine], zmax*.95, fieldLines);
    //Get coordinates of every point along field line
    for(int inLine = 0; inLine < fieldLines.size(); inLine++){
      fieldLineX = fieldLines[inLine][0];
      fieldLineY = fieldLines[inLine][1];
      fieldLineZ = fieldLines[inLine][2];
    }
    
    //Find if termination point is at pad
    //TODO: With hex geometry, this is bad criteria for if at pad!!
    int lineEnd = fieldLines.size() - 1;
    if(  (abs(fieldLines[lineEnd][0]) <= padLength/2.)
      && (abs(fieldLines[lineEnd][1]) <= padLength/2.)
      && (fieldLines[lineEnd][2] < 0.)){
        numAtPad++;
    }
    
    //Print a progress update every 10%
    int driftLineProgress = (100*(inFieldLine+1))/totalFieldLines;
    if(   (driftLineProgress % 10 == 0)
      &&  (driftLineProgress != prevDriftLine)){
      std::cout << "Driftline Progress: " << driftLineProgress << " %" << std::endl;
      prevDriftLine = driftLineProgress;
    }

  }//End field line loop
  
  std::cout << "Done " << totalFieldLines << " field lines; Determining transparency." << "\n";
  //Determine transparency
  fieldTransparency = (1.*numAtPad) / (1.*totalFieldLines);
  std::cout << "Field transparency is " << fieldTransparency <<  "." << std::endl;

  if(fieldTransparency < transparencyLimit){
    std::cout << "Warning: Field transparency is lower than the limit." << std::endl;
  }
  
  //***** Deal with files *****//
  dataFile << fieldTransparency;
  dataFile.close();

  std::cout << "****************************************\n";
  std::cout << "Done simulation: " << "\n";
  std::cout << "****************************************\n";
  std::cout << std::endl;

  return 0;

}
