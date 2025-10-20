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

  float inputField = std::atof(argv[1]); //kV/cm
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

  double xDriftVelocity=0., yDriftVelocity=0., zDriftVelocity=0.;
  double xDriftVelocityErr=0., yDriftVelocityErr=0., zDriftVelocityErr=0.;
  
  double diffusionL=0., diffusionT=0.;
  double diffusionLErr=0., diffusionTErr=0.;

  /*
  // Direct methods for diffusion parameters
  //    Requires generating a gas table.
  //    Does not output errors of values.

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
  */

  double bulkFlux, bulkDriftVelocity;
  double townsend, attatchment;
  double ionizationRate, attachmentRate, lorentzAngle;
  double bulkFluxErr, bulkDriftVelocityErr;
  double townsendErr, attatchmentErr;
  double ionizationRateErr, attachmentRateErr, lorentzAngleErr;
  double effectiveTownsend;
  std::array<double, 6> diffusionTensor;

  gasFIMS->RunMagboltz(
    //Input values:
    fieldStrength, 0., 0., 10, true,
    //Outputs:
    xDriftVelocity, yDriftVelocity, zDriftVelocity,
    bulkFlux, bulkDriftVelocity, 
    diffusionL, diffusionT,
    townsend, attatchment,
    ionizationRate, attachmentRate, lorentzAngle, 
    xDriftVelocityErr, yDriftVelocityErr, zDriftVelocityErr,
    bulkFluxErr, bulkDriftVelocityErr, 
    diffusionLErr, diffusionTErr,
    townsendErr, attatchmentErr, 
    ionizationRateErr, attachmentRateErr, lorentzAngleErr, 
    effectiveTownsend, diffusionTensor
  );

  if(xDriftVelocity != 0 || yDriftVelocity != 0){
    std::cerr << "Magboltz error: Non-zero normal drift velocity.";
  }


  std::cout << "------------------------------\n";
  std::cout << "Input field: " << inputField << " kV/cm\n";
  std::cout << "Field strength: " << fieldStrength << " V/cm\n";
  std::cout << "------------------------------\n";

  std::cout << "ElectronVelocity: " << xDriftVelocity << ", " << yDriftVelocity << ", " << std::abs(zDriftVelocity) << "\n";
  std::cout << "ElectronDiffusion: " << diffusionL << ", " << diffusionT << "\n";

  //***** Output gain value *****//	
  //create output file
  int fieldStrengthINT = fieldStrength;
  std::string dataFilename = "diffusion."+gasComposition+"."+std::to_string(fieldStrengthINT)+".dat";
  std::string dataPath = "../../Data/Diffusion/"+dataFilename;
  std::ofstream dataFile;

  //Write results to file
  dataFile.open(dataPath);

  //write some extra information
  dataFile << "// Diffusion coefficients from Magboltz.\n";
  dataFile << "//\tErrors in percentages.\n";

  dataFile << "//\tGas composition:\n";
  dataFile << gasComposition << std::endl;

  dataFile << "//\tField strength (kV/cm):\n";
  dataFile << inputField << std::endl;

  dataFile << "//\tDrift velocity (um/ns):\n";
  dataFile << std::abs(zDriftVelocity)*1e4 << std::endl;
  dataFile << zDriftVelocityErr << std::endl;

  dataFile << "//\tLongitudinal diffusion (um/cm**0.5):\n";
  dataFile << diffusionL*1e4 << std::endl;
  dataFile << diffusionLErr << std::endl;

  dataFile << "//\tTransverse diffusion (um/cm**0.5):\n";
  dataFile << diffusionT*1e4 << std::endl;
  dataFile << diffusionTErr << std::endl;

  dataFile.close();

  return 0;
}
