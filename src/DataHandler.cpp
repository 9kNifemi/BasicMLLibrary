//
//  DataHandler.cpp
//  ML_Library
//
//  Created by 9knifemi on 2024-02-29.
//
#include "DataHandler.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm> // For std::shuffle
#include <random>    // For std::default_random_engine
#include <chrono>    // For std::chrono::system_clock

DataHandler::DataHandler() {}

bool DataHandler::loadData(const std::string& filePath, bool hasHeader) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filePath << std::endl;
        return false;
    }

    std::string line;
    if (hasHeader) std::getline(file, line); // Skip header line

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        DataPoint dataPoint;
        double value;
        while (iss >> value) {
            dataPoint.features.push_back(value);
        }
        // Assume the last value is the label for simplicity
        if (!dataPoint.features.empty()) {
            dataPoint.label = dataPoint.features.back();
            dataPoint.features.pop_back(); // Remove label from features
            dataset.push_back(dataPoint);
        }
    }

    file.close();
    return true;
}

void DataHandler::normalizeFeatures() {
    for (size_t i = 0; i < dataset[0].features.size(); ++i) { // Iterate over each feature
        std::vector<double> featureColumn;
        for (const auto& dataPoint : dataset) {
            featureColumn.push_back(dataPoint.features[i]);
        }
        double mean = computeMean(featureColumn);
        double stdDev = computeStdDev(featureColumn, mean);

        for (auto& dataPoint : dataset) {
            if (stdDev != 0) { // Prevent division by zero
                dataPoint.features[i] = (dataPoint.features[i] - mean) / stdDev;
            }
        }
    }
}
double DataHandler::computeMean(const std::vector<double>& values) {
    double sum = 0.0;
    for (auto value : values) {
        sum += value;
    }
    return sum / values.size();
}
double DataHandler::computeStdDev(const std::vector<double>& values, double mean) {
    double variance = 0.0;
    for (auto value : values) {
        variance += (value - mean) * (value - mean);
    }
    variance /= values.size();
    return sqrt(variance);
}

void DataHandler::splitData(float trainingRatio) {
    shuffleData(); // Ensure the data is shuffled before splitting

    size_t splitIndex = static_cast<size_t>(dataset.size() * trainingRatio);
    for (size_t i = 0; i < splitIndex; ++i) {
        trainingData.push_back(dataset[i]);
    }
    for (size_t i = splitIndex; i < dataset.size(); ++i) {
        testData.push_back(dataset[i]);
    }
}


const std::vector<DataPoint>& DataHandler::getTrainingData() const {
    return trainingData;
}

const std::vector<DataPoint>& DataHandler::getTestData() const {
    return testData;
}

// Use this implementation for shuffleData if you haven't implemented your own
void DataHandler::shuffleData() {
    // Obtain a time-based seed:
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    shuffle(dataset.begin(), dataset.end(), std::default_random_engine(seed));
}

// Add implementations for computeMean and computeStdDev...
