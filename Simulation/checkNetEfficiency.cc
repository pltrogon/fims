/*
 * checkNetEfficiency.cc
 *
 * 
 * TODO
 * 
 */

// My includes
#include "myFunctions.h"
#include <boost/math/distributions/beta.hpp>//NOTE THIS IS AN EXTERNAL LIBRARY

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
    if(argc != 3){
        std::cerr << "Format: " << argv[0] << " <Target Efficiency> <Detection Threshold>" << std::endl;
        return 1;
    }

    double targetEfficiency = std::atof(argv[1]);
    int electronThreshold = std::atoi(argv[2]);

    const double confidenceValue = 2.;//NOTE - 1.645 for 95% confidence instead of 2 for 2-sigma

    //Randon seed
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    const double MICRONTOCM = 1e-4;

    //Read in simulation parameters
    auto simParams = readSimulationParameters();
    if(!simParams){
        return -1;
    }

    //********** Setup Simulation **********//
    std::cout << "Setting up simulation " << simParams->runNumber << "(Net efficiency)" << std::endl;

    //Gas Mixture
    MediumMagboltz* gasFIMS = initializeGas(*simParams);
    //Field map
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
    xBoundary[0] = -2.*simParams->pitch;
    xBoundary[1] = 2.*simParams->pitch;
    yBoundary[0] = -2.*simParams->pitch;
    yBoundary[1] = 2.*simParams->pitch;

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

    //Define avalanche characteristics
    int electronLimit = electronThreshold+5;
    AvalancheMicroscopic* avalancheE = new AvalancheMicroscopic;
    avalancheE->SetSensor(sensorFIMS);
    avalancheE->EnableAvalancheSizeLimit(electronLimit);
    {
        SilenceCerr guard;
        avalancheE->EnablePlotting(nullptr, 10);//For velocity vector
    }

    //Deafult initial electron parameters
    double x0 = 0., y0 = 0., z0 = 0.75*simParams->holeRadius;
    double t0 = 0.;//ns
    double e0 = 0.1;//eV (Garfield is weird when this is 0.)
    double dx0 = 0., dy0 = 0., dz0 = 0.;//No velocity

    //Set up some data variables
    int numInitialElectrons = 0;//Number of initial electrons generated 
    int numTotalTrials = 0;//Number of electrons populated
    int numAboveThreshold = 0;
    int numCollected = 0;
    int numHitGrid = 0;
    int numFailure = 0;

    //Statistics variables
    double detectionEff = 0., collectionEff = 0., netEfficiency = 0.; 
    double upperDetectionLimit = 1., lowerDetectionLimit = 0.;
    double upperCollectionLimit = 1., lowerCollectionLimit = 0.;
    double netEfficiencyLow = 0., netEfficiencyHigh = 1.;
    double netEfficiencyErr = 0.;

    double lowerCollection = 0., upperCollection = 0., meanCollection = 0.;
    double lowerDetection = 0., upperDetection = 0., meanDetection = 0.;

    //Initial loop control
    int numInBunch = 500;//Always do at least 500 avalanches first
    bool runAvalanche = true, isEfficienct = false;

    double cellLength = simParams->pitch/std::sqrt(3.);

    //TEstingStats
    int num7 = 0, num5 = 0, num1 = 0;

    std::cout << "Beginning avalanches..." << std::endl;
    //Run avalanches in bunches
    while(runAvalanche && numInitialElectrons < simParams->numAvalanche){
        for(int inAvalanche=0; inAvalanche < numInBunch; inAvalanche++){
            numInitialElectrons++;

            //Random xy on plane
            auto [sampleX, sampleY] = randomXYInHexagon(cellLength);
            double curX = sampleX, curY = sampleY, curZ = z0;
            double curTime = t0;
            double curEnergy = e0;
            double curDx = 0., curDy = 0., curDz = 0.;

            //Parameters to grab electron data
            double xi, yi, zi, ti, Ei;
            double xf, yf, zf, tf, Ef;
            int exitStatus;

            bool repopulate = true;
            while(repopulate){
                //Populate with an electron
                numTotalTrials++;
                avalancheE->AvalancheElectron(
                    curX, curY, curZ, 
                    curTime, curEnergy, 
                    curDx, curDy, curDz
                );

                int numAvalancheElectrons = avalancheE->GetNumberOfElectronEndpoints();

                //Ensure electron didnt disappear. Reinitialize if so. 
                if(numAvalancheElectrons == 0){
                    std::cerr << "Error: No electrons in avalanche - Restarting." << std::endl;
                    std::cerr << "\tError at (" << curX << ", " << curY << ", " << curZ << ")" << std::endl;
                    numFailure++;
                    exitStatus = -7;//Treat it the same as an attachment
                }
                else{
                    avalancheE->GetElectronEndpoint(0, xi, yi, zi, ti, Ei, xf, yf, zf, tf, Ef, exitStatus);
                }

                //Check if above threshold for detection efficiency. Assume it is collected.
                if(numAvalancheElectrons >= electronThreshold){
                    numAboveThreshold++;
                    numCollected++;
                    break;
                }

                //Only check where electron ends if there is only 1 (Any larger is assumed to be collected)
                if(numAvalancheElectrons == 1){
                    switch(exitStatus){

                        // Electron attached to gas molecule - Restart with initial electron
                        //WARNING - This may cause an infinite loop. Consider max attempts if becomes an issue
                        case -7: {
                            curX = sampleX, curY = sampleY, curZ = z0;
                            curTime = t0;
                            curEnergy = e0;
                            curDx = 0., curDy = 0., curDz = 0.;
                            num7++;
                            break;
                        }

                        // Electron leaves drift medium (Hits Grid/Pad/Dielectric)
                        case -5: {
                            if(zf < -1.*simParams->gridThickness){
                                numCollected++;
                            }
                            else{
                                numHitGrid++;
                            }
                            num5++;
                            repopulate = false;
                            break;
                        }

                        // Electron leaves the simulation volume - Shift it back
                        //Determine which boundary was hit and shift by pitch to opposite side
                        case -1: {
                            //Shift x or y by pitch and keep z (Use some buffer)
                            curX = std::abs(xf) >= .99*2.*simParams->pitch ? -1.*std::copysign(simParams->pitch, xf) : xf;
                            curY = std::abs(yf) >= .99*2.*simParams->pitch ? -1.*std::copysign(simParams->pitch, yf) : yf;
                            curZ = zf;

                            //Get direction vector from 2nd-last and last points along drift
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

                            curTime = tf;
                            curEnergy = Ef;

                            num1++;
                            break;
                        }

                        default:
                            std::cerr << "Error: Unexpected electron endpoint status (" << exitStatus << ")" << std::endl;
                            return -1;

                    }//End of electron endpoint switch
                }

            }//End of single avalanche trial

        }//end of avalanche bunch loop
        std::cout << "Done " << numInitialElectrons << " trials." << std::endl;

        numInBunch = 100;//do bunches of 100 after first iteration

        //Calculate efficiencies - Bayesian stats (Depreciated?)
        collectionEff = (double)(numCollected+1.) / (numInitialElectrons+2.);
        detectionEff = (double)(numAboveThreshold+1.) / (numCollected+2.);

        netEfficiency = detectionEff*collectionEff;

        // *** Check efficiency ***
        const double pLower = 0.02275;
        const double pUpper = 0.97725;

        //Collection Efficiency
        boost::math::beta_distribution<> distCollection(
            numCollected + 1., 
            (numInitialElectrons - numCollected) + 1.
        );

        meanCollection  = boost::math::mean(distCollection);
        lowerCollection = boost::math::quantile(distCollection, pLower);
        upperCollection = boost::math::quantile(distCollection, pUpper);

        //Detection Efficiency
        if(numCollected > 0){
            boost::math::beta_distribution<> distDetection(
                numAboveThreshold + 1., 
                (numCollected - numAboveThreshold) + 1.
            );
            
            meanDetection  = boost::math::mean(distDetection);
            lowerDetection = boost::math::quantile(distDetection, pLower);
            upperDetection = boost::math::quantile(distDetection, pUpper);
        }

        //Net efficiency
        netEfficiency = meanCollection*meanDetection;
        netEfficiencyLow = lowerCollection*lowerDetection;
        netEfficiencyHigh = upperCollection*upperDetection;

        //Efficiency exceeds or excludes target within confidence
        if(netEfficiencyHigh < targetEfficiency || netEfficiencyLow >= targetEfficiency){
            runAvalanche = false;
        }


    }//End of all avalanches


    //***** Output efficiency value *****//	
    //create output file
    std::string dataFilename = "netEfficiencyFile.dat";
    std::string dataPath = "../../Data/"+dataFilename;
    std::ofstream dataFile;

    //Write results to file
    dataFile.open(dataPath);
    if(!dataFile.is_open()){
        std::cerr << "Error: Could not open file: " << dataPath << std::endl;
    }

    //write some extra information
    dataFile << "// Finding net efficiency for run: " << simParams->runNumber << "\n";
    dataFile << "// Field Ratio: " << simParams->fieldRatio << "\n";
    dataFile << "// Total avalanches: " << numInitialElectrons << " (of " << simParams->numAvalanche << ")\n";
    dataFile << "// Electron threshold: " << electronThreshold << "\n";
    dataFile << "// Num Detected: " << numAboveThreshold << "\n";
    dataFile << "// Num Collected: " << numCollected << "\n";

    dataFile << "// Collection Efficiency: " << meanCollection << "\n";
    dataFile << "// Collection Efficiency Range: (" << lowerCollection << ", " << upperCollection << ")\n";
    dataFile << "// Detection Efficiency: " << meanDetection << "\n";
    dataFile << "// Detection Efficiency Range: (" << lowerDetection << ", " << upperDetection << ")\n";

    //include convergence criteria
    dataFile << "// Stop condition:\n";
    if(runAvalanche){
        dataFile << "DID NOT CONVERGE\n";
    }
    else{
        dataFile << "CONVERGED\n";
    }

    //output efficiency
    dataFile << "// Net:\n" << netEfficiency << "\n" << (netEfficiency-netEfficiencyLow) << "\n" << (netEfficiencyHigh-netEfficiency) << std::endl;

    dataFile.close();

    return 0;

}
