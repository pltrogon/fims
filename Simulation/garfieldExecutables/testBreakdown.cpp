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
//Garfield includes
#include "Garfield/MediumMagboltz.hh"

using namespace Garfield;

int main() {
    // Setup the Gas Mixture
    MediumMagboltz gas;
    gas.SetComposition(
        "Ar", 95., 
        "CF4", 3., 
        "iC4H10", 2.
    );
    gas.SetTemperature(293.15);
    gas.SetPressure(760.);

    // Define the range of Electric Fields
    const int nE = 51;
    const double eMin = 1000.0;  // V/cm
    const double eMax = 100000.0; // V/cm
    gas.SetFieldGrid(eMin, eMax, nE, true);

    // Generate the gas table
    const int nColl = 7;
    gas.GenerateGasTable(nColl);

    // Calculate Breakdown points
    const double gamma = 0.1; //Second Townsend coefficient Aluminum ~0.1
    const double p = 760.0;

    std::cout << "pd [Torr-cm], Vb [V]" << std::endl;

    std::vector<double> efields;
    std::vector<double> bfields;
    std::vector<double> angles;
    gas.GetFieldGrid(efields, bfields, angles);


    //***** Output transparency value *****//	
	//create output file
    std::string dataFilename = "testBreakdown.dat";
    std::string dataPath = "../../Data/"+dataFilename;
    std::ofstream dataFile;

    dataFile.open(dataPath, std::ios::app);
    if(!dataFile.is_open()){
        std::cerr << "Error: Could not open file: " << dataPath << std::endl;
    }    

    for (size_t i = 0; i < efields.size(); ++i) {
        double field = efields[i]; 

        double logAlpha = 0.;
        if (gas.GetElectronTownsend(i, 0, 0, logAlpha)) {
            double alpha = exp(logAlpha);
            
            if (alpha > 1e-10) {
                // Townsend Breakdown Criterion
                double d = log(1.0 + 1.0 / gamma) / alpha;
                double Vb = field * d;
                double pd = p * d;
                dataFile << pd << ", " << Vb << std::endl;
            }
        }
    }


    dataFile.close();

    return 0;
}
