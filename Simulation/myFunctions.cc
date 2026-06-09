#include "myFunctions.hh"
#include <algorithm>

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
 * @brief Generates a random (x,y) point uniformly distributed within a hexagon centered at the origin with the specified side length.
 * @param sideLength The length of each side of the hexagon.
 * 
 * @return A pair of doubles representing the (x,y) coordinates of the random point.
 */
std::pair<double, double> randomXYInHexagon(double sideLength) {

    const double sqrt3 = std::sqrt(3.0);
    const double inRadius = sqrt3 * sideLength / 2.;
    const double outRadius = sideLength;

    thread_local static std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    while(true) {
        // Uniform sample in box
        double sampleX = dist(rng) * outRadius;
        double sampleY = dist(rng) * inRadius;

        // Check if in hexagon (use symmetry of Q1)
        double absX = std::fabs(sampleX);
        double absY = std::fabs(sampleY);
        if(absX <= sideLength - absY/sqrt3){
            return {sampleX, sampleY};
        }
    }
}

/**
 * @brief Utility to temporarily silence std::cerr.
 */
SilenceCerr::SilenceCerr() {
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

SilenceCerr::~SilenceCerr() {
    // Restore the original buffer on destruction
    if (m_oldBuffer) {
        std::cerr.rdbuf(m_oldBuffer);
    }
}

/**
 * Initializes a Garfield++ gas mixture with the provided parameters.
 * 
 * @param params SimulationParameters struct containing gas composition and settings
 * @return Pointer to initialized MediumMagboltz object (caller responsible for deletion)
 */
Garfield::MediumMagboltz* initializeGas(const SimulationParameters& params) {

    std::cout << "Initializing gas mixture..." << std::endl;

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

    std::cout << "Gas mixture initialized." << std::endl;
    
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
    while(std::getline(std::cin, line)) {
        jsonInput += line;
    }

    if(jsonInput.empty()) {
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
        params.initialZFraction = params_json["initialZFraction"].get<double>();
        
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

/**
 * @brief Calculates the mean efficiency and its asymmetric error bars using a Beta distribution.
 * This function estimates the probability of success (efficiency) based on a given 
 * number of successes and trials. It uses a Bayesian approach with a uniform prior.
 * 
 * @param nSuccess The number of successful outcomes recorded.
 * @param nTotal The total number of trials conducted.
 * @return Efficiency results {meanValue, lowError, highError, minValue, maxValue}
 */
EfficiencyResults calculateEfficiencyStats(int nSuccess, int nTotal) {
    
    // 2-sigma confidence intervals
    const double pLower = 0.02275; 
    const double pUpper = 0.97725;

    // Ensure data exists
    if (nTotal <= 0) {
        return {0.0, 0.0, 1.0, 0.0, 1.0};
    }

    // Define a beta distribution based on the successes and total trials.
    boost::math::beta_distribution<> betaDistribution(nSuccess + 1., nTotal - nSuccess + 1.);

    // Calculate stats
    double meanValue = boost::math::mean(betaDistribution);
    double lowerLimit = boost::math::quantile(betaDistribution, pLower);
    double upperLimit = boost::math::quantile(betaDistribution, pUpper);

    // Find errors
    double lowError = meanValue - lowerLimit;
    double highError = upperLimit - meanValue;

    return {meanValue, lowError, highError, lowerLimit, upperLimit};
}

/**
 * @brief Converts a string to a GeometryMode enum value.
 * @param str The string to convert (case-insensitive).
 * @return The corresponding GeometryMode enum value.
 */
GeometryMode stringToGeometryMode(std::string str) {
    std::transform(str.begin(), str.end(), str.begin(), ::tolower);

    if(str == "fims") {
        return GeometryMode::FIMS;
    }
    if(str == "fimssurrounding") {
        return GeometryMode::FIMSSurrounding;
    }
    
    return GeometryMode::Unknown;
}

/**
 * @brief Converts a string to an EfficiencyMode enum value.
 * @param str The string to convert (case-insensitive).
 * @return The corresponding EfficiencyMode enum value.
 */
EfficiencyMode stringToEfficiencyMode(std::string str) {
    std::transform(str.begin(), str.end(), str.begin(), ::tolower);

    if(str == "net") {
        return EfficiencyMode::Net;
    }
    if(str == "detection") {
        return EfficiencyMode::Detection;
    }
    if(str == "collection") {
        return EfficiencyMode::Collection;
    }
    
    return EfficiencyMode::Unknown;
}
