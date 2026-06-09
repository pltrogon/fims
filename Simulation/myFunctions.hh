#ifndef MY_FUNCTIONS_H
#define MY_FUNCTIONS_H

#include <fstream>
#include <string>
#include <map>
#include <iostream>
#include <optional>
#include <nlohmann/json.hpp>
#include <random>
#include <cmath>
#include <utility>
#include <functional>

#include "Garfield/ComponentElmer.hh"
#include "Garfield/AvalancheMicroscopic.hh"
#include "Garfield/MediumMagboltz.hh"
#include "Garfield/Medium.hh"
#include "Garfield/AvalancheMC.hh"
#include "Garfield/Sensor.hh"
#include "Garfield/DriftLineRKF.hh"
#include "Garfield/ViewDrift.hh"
#include "Garfield/ViewSignal.hh"

#include <boost/math/distributions/beta.hpp>

using json = nlohmann::json;

// Conversion constant
const double MICRONTOCM = 1e-4;

// Struct to hold all simulation parameters
// Compare to simulationClass for consistency
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
    double initialZFraction;
    
    // Gas parameters
    double gasCompAr;
    double gasCompCO2;
    double gasCompCF4;
    double gasCompIsobutane;
    double gasPenning;
};

// Struct to hold efficiency parameters
struct EfficiencyResults {
    double meanValue;
    double lowError;
    double highError;
    double minValue;
    double maxValue;
};

/**
 * Retrieves the current git version/hash.
 * 
 * @return std::string containing git version, or "UNKNOWN VERSION" if retrieval fails
 */
std::string getGitVersion();

/**
 * @brief Generates a random (x,y) point uniformly distributed within a hexagon centered at the origin with the specified side length.
 * @param sideLength The length of each side of the hexagon.
 * 
 * @return A pair of doubles representing the (x,y) coordinates of the random point.
 */
std::pair<double, double> randomXYInHexagon(double sideLength);

/**
 * @brief Utility to temporarily silence std::cerr.
 */
class SilenceCerr {
public:
    SilenceCerr();
    ~SilenceCerr();

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
Garfield::MediumMagboltz* initializeGas(const SimulationParameters& params);

/**
 * Reads and parses simulation parameters from stdin as JSON.
 * 
 * @return SimulationParameters if successful, empty optional if failed
 */
std::optional<SimulationParameters> readSimulationParameters();

/**
 * @brief Calculates the mean efficiency and its asymmetric error bars using a Beta distribution.
 * This function estimates the probability of success (efficiency) based on a given 
 * number of successes and trials. It uses a Bayesian approach with a uniform prior.
 * 
 * @param nSuccess The number of successful outcomes recorded.
 * @param nTotal The total number of trials conducted.
 * @return Efficiency results {meanValue, lowError, highError, minValue, maxValue}
 */
EfficiencyResults calculateEfficiencyStats(int nSuccess, int nTotal);

// Geometry mode enumeration and conversion
enum class GeometryMode {
    FIMS,
    FIMSSurrounding,
    Unknown
};

/**
 * @brief Converts a string to a GeometryMode enum value.
 * @param str The string to convert (case-insensitive).
 * @return The corresponding GeometryMode enum value.
 */
GeometryMode stringToGeometryMode(std::string str);

// Efficiency mode enumeration and conversion
enum class EfficiencyMode {
    Net,
    Detection,
    Collection,
    Unknown
};

/**
 * @brief Converts a string to an EfficiencyMode enum value.
 * @param str The string to convert (case-insensitive).
 * @return The corresponding EfficiencyMode enum value.
 */
EfficiencyMode stringToEfficiencyMode(std::string str);

#endif // MY_FUNCTIONS_H
