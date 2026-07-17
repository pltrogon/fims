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

// Garfield includes
#include "Garfield/MediumMagboltz.hh"

//My includes
#include "myFunctions.hh"

using namespace Garfield;

int main(){
    
    //***** Simulation Parameters *****//
    auto simParams = readSimulationParameters();
    if(!simParams){
        return -1;
    }
    // Setup the Gas Mixture
    MediumMagboltz* gasFIMS = initializeGas(*simParams); 


    // ----- Define Gamma here ----- //
    const double gamma = 1e-4;


    // Define the range of Electric Fields
    const int numFields = 100;
    const double eMin = 1e3;// 1 kV/cm
    const double eMax = 2e5;// 200 kV/cm
    gasFIMS->SetFieldGrid(eMin, eMax, numFields, true); // Logarithmic spacing

    // Generate the gas table
    std::cout << "Generating gas table..." << std::endl;
    const int nColl = 3;
    gasFIMS->GenerateGasTable(nColl);

    //Get fields
    std::cout << "Getting fields..." << std::endl;
    std::vector<double> eFields;
    std::vector<double> bFields;
    std::vector<double> angles;
    gasFIMS->GetFieldGrid(eFields, bFields, angles);

    // Create output file
    std::string dataFilename = "gasBreakdown.dat";
    std::string dataPath = "../../Data/" + dataFilename;
    std::ofstream dataFile;
    dataFile.open(dataPath, std::ios::trunc); // Overwrite fresh

    if (!dataFile.is_open()) {
        std::cerr << "Error: Could not open file: " << dataPath << std::endl;
        return -1;
    }

    // Header for readability
    dataFile << "paschenX_Torrcm,paschenY_V,dist_um,breakdownField_kVcm" << std::endl;

    //Get the data
    std::cout << "Getting coeficencts..." << std::endl;
    double gasPressure = 760.0;//torr
    double logGammaTerm = std::log(1.0 + 1.0/gamma);
    for(int i = 0; i < numFields; i++){
        //Get E field and townsend coeff
        double inField = eFields[i];
        double logAlpha = 0.0;        

        if(gasFIMS->GetElectronTownsend(i, 0, 0, logAlpha)){

            if(logAlpha < -5.0){continue;}

            double alpha = std::exp(logAlpha);
            double dist_cm = logGammaTerm / alpha;
            double dist_um = dist_cm*1e4;
            double breakdownV = inField*dist_cm;
            double field_kV = inField/1000.;

            //Paschen parameters
            double paschenX = dist_cm*gasPressure;
            double paschenY = breakdownV;

            dataFile << paschenX << "," << paschenY << "," << dist_um << "," << field_kV << std::endl;
        }
    }

    dataFile.close();
    delete gasFIMS;
    return 0;
}
