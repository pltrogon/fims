#ifndef MY_FUNCTIONS_H
#define MY_FUNCTIONS_H

#include <fstream>
#include <string>
#include <map>
#include <iostream>
#include <optional>
#include <nlohmann/json.hpp>

#include "Garfield/ComponentElmer.hh"
#include "Garfield/AvalancheMicroscopic.hh"
#include "Garfield/MediumMagboltz.hh"
#include "Garfield/Medium.hh"
#include "Garfield/AvalancheMC.hh"
#include "Garfield/Sensor.hh"
#include "Garfield/DriftLineRKF.hh"
#include "Garfield/ViewDrift.hh"
#include "Garfield/ViewSignal.hh"

using json = nlohmann::json;

// Conversion constant
const double MICRONTOCM = 1e-4;

// Struct to hold all simulation parameters
// Compare to simulationClass for consistancy
struct SimulationParameters {
    // Geometry parameters
    double padLength;
    double pitch;
    double gridStandoff;
    double gridThickness;
    double holeRadius;
    double cathodeHeight;
    double thicknessSiO2;
    double pillarRadius;
    
    // Field parameters
    double driftField;
    double fieldRatio;
    int numFieldLine;
    
    // Simulation parameters
    int runNumber;
    int numAvalanche;
    int avalancheLimit;
    int numInputs;
    
    // Gas parameters
    double gasCompAr;
    double gasCompCO2;
    double gasCompCF4;
    double gasCompIsobutane;
    double gasPenning;
};

/**
 * Retrieves the current git version/hash.
 * 
 * @return std::string containing git version, or "UNKNOWN VERSION" if retrieval fails
 */
std::string getGitVersion() {
    std::string gitVersion = "UNKNOWN VERSION";
    const char* getGitCommand = "git describe --tags --always --dirty 2>/dev/null";
    
    FILE* pipe = popen(getGitCommand, "r");
    if (!pipe) {
        return gitVersion;
    }
    
    char gitBuffer[128];
    std::string gitOutput = "";
    while (fgets(gitBuffer, sizeof(gitBuffer), pipe) != NULL) {
        gitOutput += gitBuffer;
    }
    int gitStatus = pclose(pipe);
    
    if (gitStatus == 0 && !gitOutput.empty()) {
        if (gitOutput.back() == '\n') {
            gitOutput.pop_back();
        }
        gitVersion = gitOutput;
    }
    
    return gitVersion;
}

/**
 * @brief Utility to temporarily silence std::cerr.
 */
class SilenceCerr {
public:
    SilenceCerr() {
        // Platform detection for the "null" device
        #ifdef _WIN32
            const char* nullDevice = "nul";
        #else
            const char* nullDevice = "/dev/null";
        #endif

        m_nullStream.open(nullDevice);
        if (m_nullStream.is_open()) {
            // Save the old buffer and redirect cerr
            m_oldBuffer = std::cerr.rdbuf(m_nullStream.rdbuf());
        } else {
            m_oldBuffer = nullptr;
        }
    }

    ~SilenceCerr() {
        // Restore the original buffer on destruction
        if (m_oldBuffer) {
            std::cerr.rdbuf(m_oldBuffer);
        }
    }

    // Disable copying to prevent multiple objects fighting over the same buffer
    SilenceCerr(const SilenceCerr&) = delete;
    SilenceCerr& operator=(const SilenceCerr&) = delete;

private:
    std::streambuf* m_oldBuffer;
    std::ofstream m_nullStream;
};

/**
 * Initializes a Garfield++ gas mixture with the provided parameters.
 * 
 * @param params SimulationParameters struct containing gas composition and settings
 * @return Pointer to initialized MediumMagboltz object (caller responsible for deletion)
 */
Garfield::MediumMagboltz* initializeGas(const SimulationParameters& params) {
    Garfield::MediumMagboltz* gas = new Garfield::MediumMagboltz();
    
    // Set gas composition (converting from percentage back to fraction for Garfield)
    gas->SetComposition(
        "ar", params.gasCompAr / 100.0,
        "co2", params.gasCompCO2 / 100.0,
        "cf4", params.gasCompCF4 / 100.0,
        "iC4H10", params.gasCompIsobutane / 100.0
    );
    
    // Enable Penning transfer
    {
        SilenceCerr guard;
        gas->EnablePenningTransfer(params.gasPenning, 0.0, "ar");
    }
    
    // Set STP gas parameters
    double gasTemperature = 293.15;  // K
    double gasPressure = 760.0;       // torr
    int maxElectronE = 200;           // eV
    
    gas->SetTemperature(gasTemperature);
    gas->SetPressure(gasPressure);
    gas->SetMaxElectronEnergy(maxElectronE);
    gas->Initialise(true);
    
    // Load ion mobilities
    const std::string path = std::getenv("GARFIELD_INSTALL");
    const std::string posIonPath = path + "/share/Garfield/Data/IonMobility_Ar+_Ar.txt";
    const std::string negIonPath = path + "/share/Garfield/Data/IonMobility_CF4+_CF4.txt";
    gas->LoadIonMobility(posIonPath);
    gas->LoadNegativeIonMobility(negIonPath);
    
    return gas;
}

/**
 * Reads and parses simulation parameters from stdin as JSON.
 * 
 * @return SimulationParameters if successful, empty optional if failed
 */
std::optional<SimulationParameters> readSimulationParameters() {
    std::string jsonInput;
    std::string line;
    
    // Read all lines from stdin until EOF
    while(std::getline(std::cin, line)){
        jsonInput += line;
    }

    if(jsonInput.empty()){
        std::cerr << "Error: No parameters received via stdin." << std::endl;
        return std::nullopt;
    }

    try {
        json params_json = json::parse(jsonInput);
        SimulationParameters params;
        
        // Geometry parameters (converting from microns to cm)
        params.padLength = params_json["padLength"].get<double>() * MICRONTOCM;
        params.pitch = params_json["pitch"].get<double>() * MICRONTOCM;
        params.gridStandoff = params_json["gridStandoff"].get<double>() * MICRONTOCM;
        params.gridThickness = params_json["gridThickness"].get<double>() * MICRONTOCM;
        params.holeRadius = params_json["holeRadius"].get<double>() * MICRONTOCM;
        params.cathodeHeight = params_json["cathodeHeight"].get<double>() * MICRONTOCM;
        params.thicknessSiO2 = params_json["thicknessSiO2"].get<double>() * MICRONTOCM;
        params.pillarRadius = params_json["pillarRadius"].get<double>() * MICRONTOCM;

        // Field parameters
        params.driftField = params_json["driftField"].get<double>();
        params.fieldRatio = params_json["fieldRatio"].get<double>();
        params.numFieldLine = params_json["numFieldLine"].get<int>();

        // Simulation parameters
        params.runNumber = params_json["runNumber"].get<int>();
        params.numAvalanche = params_json["numAvalanche"].get<int>();
        params.avalancheLimit = params_json["avalancheLimit"].get<int>();
        
        // Gas parameters (converting from fraction to percentage)
        params.gasCompAr = params_json["gasCompAr"].get<double>() * 100.0;
        params.gasCompCO2 = params_json["gasCompCO2"].get<double>() * 100.0;
        params.gasCompCF4 = params_json["gasCompCF4"].get<double>() * 100.0;
        params.gasCompIsobutane = params_json["gasCompIsobutane"].get<double>() * 100.0;
        params.gasPenning = params_json["gasPenning"].get<double>();
        
        params.numInputs = 18;  // Number of parameters

        return params;
    } catch(const std::exception& e) {
        std::cerr << "Error: Failed to parse JSON parameters from stdin: " << e.what() << std::endl;
        return std::nullopt;
    }
}



#endif // MY_FUNCTIONS_H
