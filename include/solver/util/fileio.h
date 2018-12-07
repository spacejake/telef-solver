#pragma once

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace telef::solver::io {

    template <class DATA_TYPE>
    inline void parseCSVFile(std::vector<DATA_TYPE> &data, std::string file) {
        //vector<vector<string> > data;

        std::ifstream infile(file);
        if (!infile.is_open())
        {
            std::cout << "File Not Found: "<< file << std::endl;
            throw std::runtime_error((std::string("File Not Found: ") + file).c_str());
        }
        std::string line;

        //  Read the file
        while (getline(infile, line)) {
            std::istringstream ss(line);
            std::vector<std::string> record;

            std::string raw_data;
            while (getline(ss, raw_data, ',')) {
                std::istringstream ss_data(raw_data);

                DATA_TYPE record;

                while( ss_data >> record || !ss_data.eof()) {
                    if (!ss_data.fail()) {
                        data.push_back(record);
                        break;
                    } else {
                        ss_data.clear();

                        std::string dummy;
                        ss_data >> dummy;

                        std::cout << "Invalid data found in file: "<< dummy << std::endl;
                        throw std::runtime_error(
                                (std::string("Exception: Invalid data found in file.") + dummy).c_str());
//                        ss_data.clear(); // unset failbit
//                        ss_data.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                    }
                }
            }
        }

    }
}
