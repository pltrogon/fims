#include "Garfield/MediumMagboltz.hh"

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

int main(int argc, char* argv[]) {

  if((argc != 3) && (argc != 5)){
    std::cerr << "Format: " << argv[0] << "<InputField (kV/cm)> <GasComposition> (Optional: <ArComp> <CO2Comp>)" << std::endl;
    return 1;
  }

  int inputField = std::atoi(argv[1]); //kV/cm
  double fieldStrength = inputField*1e3;//V/cm

  MediumMagboltz* gasFIMS = new MediumMagboltz();

  std::string gasComp = argv[2];
  std::string gasComposition;
  //Gas composition
  if(gasComp == "T2K"){
    gasComposition = gasComp;
    gasFIMS->SetComposition(
      "ar", 95.0,
      "cf4", 3.0, 
      "iC4H10", 2.0
  );
  }
  else if(gasComp == "ArCO2"){
    int gasCompAr = std::atoi(argv[3]);
    int gasCompCO2 = std::atoi(argv[4]);
    if(gasCompAr + gasCompCO2 != 100){
      std::cerr << "Error: Gas composition of ArCO2 is not 100%." << std::endl;
      return 1;
    }

    gasComposition = gasComp+"-"+std::to_string(gasCompAr)+"-"+std::to_string(gasCompCO2);
    gasFIMS->SetComposition(
      "ar", gasCompAr, 
      "co2", gasCompCO2
    );
  }

  //gas parameters:
  gasFIMS->SetTemperature(293.15);
  gasFIMS->SetPressure(760.);
  gasFIMS->SetMaxElectronEnergy(200);
  gasFIMS->Initialise(true);

  std::vector<double> eField = {fieldStrength};
  std::vector<double> bField = {0.};
  std::vector<double> angles = {0.};

  std::cout << "Generating gas tables...\n";
  gasFIMS->SetFieldGrid(eField, bField, angles);
  gasFIMS->GenerateGasTable(10);


  double xDriftVelocity, yDriftVelocity, zDriftVelocity;
  double diffusionL, diffusionT;

  gasFIMS->ElectronVelocity(
    0, 0, fieldStrength, 
    0, 0, 0, 
    xDriftVelocity, yDriftVelocity, zDriftVelocity
  );  
  gasFIMS->ElectronDiffusion(
    0., 0., fieldStrength, 
    0., 0., 0., 
    diffusionL, 
    diffusionT
  );

  /*
  //Via runMagboltz
  double vx, vy, wv, wr;
  double alpha, eta, riontof, ratttof, lor;
  double vxerr, vyerr, vzerr, wverr, wrerr;
  double alphaerr, etaerr, riontoferr, ratttoferr, lorerr, alphatof;
  std::array<double, 6> difftens;

  // Drift field
  double diffL, diffT, driftVelocity, diffLErr, diffTErr;


  gasFIMS->RunMagboltz(
    fieldStrength, 0., 0., 1, true,
    vx, vy, driftVelocity, wv, wr, 
    diffL, diffT,
    alpha, eta, riontof, ratttof, lor, 
    vxerr, vyerr, vzerr, wverr, wrerr, 
    diffLErr, diffTErr,
    alphaerr, etaerr, riontoferr, ratttoferr, lorerr, alphatof,
    difftens
  );
  */

  std::cout << "------------------------------\n";
  std::cout << "Input field: " << inputField << " kV/cm\n";
  std::cout << "Field strength: " << fieldStrength << " V/cm\n";
  std::cout << "------------------------------\n";

  std::cout << "ElectronVelocity: " << xDriftVelocity << ", " << yDriftVelocity << ", " << std::abs(zDriftVelocity) << "\n";
  std::cout << "ElectronDiffusion: " << diffusionL << ", " << diffusionT << "\n";

  //***** Output gain value *****//	
  //create output file
  std::string dataFilename = "diffusion."+gasComposition+"."+std::to_string(inputField)+".dat";
  std::string dataPath = "../../Data/Diffusion/"+dataFilename;
  std::ofstream dataFile;

  //Write results to file
  dataFile.open(dataPath);

  //write some extra information
  dataFile << "// Diffusion coefficients from Magboltz.\n";
  dataFile << "//\tGas composition: " << gasComposition << "\n";
  dataFile << "//\tField strength: " << inputField << " kV/cm\n";

  dataFile << "// Drift velocity (um/ns):\n";
  dataFile << std::abs(zDriftVelocity)*1e4 << std::endl;

  dataFile << "// Longitudinal diffusion (um/cm**0.5):\n";
  dataFile << diffusionL*1e4 << std::endl;

  dataFile << "// Transverse diffusion (um/cm**0.5):\n";
  dataFile << diffusionT*1e4 << std::endl;

  dataFile.close();

  return 0;
}
