#ifndef SILENCE_CONSOLE_H
#define SILENCE_CONSOLE_H

#include <iostream>
#include <fstream>
#include <string>

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

#endif // SILENCE_CONSOLE_H
