/*
 * particle_filter.h
 *
 * 2D particle filter class.
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#ifndef PARTICLE_FILTER_H_
#define PARTICLE_FILTER_H_

#include "helper_functions.h"

struct Particle {
	int id;
	double x;
	double y;
	double theta;
	double weight;
	std::vector<int> associations;
	std::vector<double> sense_x;
	std::vector<double> sense_y;
};



class ParticleFilter {

	// Number of particles to draw
	int num_particles;

	// Flag, if filter is initialized
	bool is_initialized;

public:

	// Set of current particles
	std::vector<Particle> particles;

	// Constructor
	// @param num_particles Number of particles
	ParticleFilter() : num_particles(0), is_initialized(false) {}

	// Destructor
	~ParticleFilter() {}

	/**
	 * init Initializes particle filter by initializing particles to Gaussian
	 *   distribution around first position and all the weights to 1.
	 * @param x Initial x position [m] (simulated estimate from GPS)
	 * @param y Initial y position [m]
	 * @param theta Initial orientation [rad]
	 * @param std[] Array of dimension 3 [standard deviation of x [m], standard deviation of y [m]
	 *   standard deviation of yaw [rad]]
	 */
	void init(double x, double y, double theta, double std[]);

	/**
	 * prediction Predicts the state for the next time step
	 *   using the process model.
	 * @param delta_t Time between time step t and t+1 in measurements [s]
	 * @param std_pos[] Array of dimension 3 [standard deviation of x [m], standard deviation of y [m]
	 *   standard deviation of yaw [rad]]
	 * @param velocity Velocity of car from t to t+1 [m/s]
	 * @param yaw_rate Yaw rate of car from t to t+1 [rad/s]
	 */
	void prediction(double delta_t, double std_pos[], double velocity, double yaw_rate);

	// add observation [obs] to particle [p] transforming coordinates to the particle's space, add converted
	// observation to [p_observations]
	void addObservation(Particle& p, const LandmarkObs& obs, const double& sensor_range, std::vector<LandmarkObs>& p_observations);

	/**
	 * updateWeights Updates the weights for each particle based on the likelihood of the
	 *   observed measurements.
	 * @param sensor_range Range [m] of sensor
	 * @param std_landmark[] Array of dimension 2 [Landmark measurement uncertainty [x [m], y [m]]]
	 * @param observations Vector of landmark observations
	 * @param map Map class containing map landmarks
	 */
	void updateWeights(double sensor_range, double std_landmark[], const std::vector<LandmarkObs>& observations,
					   const Map& map_landmarks);

	/**
	 * resample Resamples from the updated set of particles to form
	 *   the new set of particles.
	 */
	void resample();

	// calculate particle weight from observations and associated nearest landmarks, with sigma uncertainty for landmark positions
	double calculateParticleWeight(const std::vector<LandmarkObs>& pObservations, const std::vector<LandmarkObs>& pLandmarks, const double sigma[]);

	/*
	 * Set a particles list of associations, along with the associations calculated world x,y coordinates
	 * This can be a very useful debugging tool to make sure transformations are correct and assocations correctly connected
	 */
	Particle SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y);

	std::string getAssociations(Particle best);
	std::string getSenseX(Particle best);
	std::string getSenseY(Particle best);

	/**
	 * initialized Returns whether particle filter is initialized yet or not.
	 */
	const bool initialized() const {
		return is_initialized;
	}

private:
	// constrain angle 0 .. 2*M_PI range
	inline double constrainRadian(double x) {
		double M_2PI = M_PI * 2;
		if (x > M_2PI)
			return fmod(x, M_2PI);
		else if (x < 0)
			return fmod(x, M_2PI) + M_2PI;
		return x;
	}

	// get distance between two points, (x1,y1) and (x2,y2)
	inline double getDistance(const double& x1, const double& y1, const double& x2, const double& y2) {
		return sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2));
	}

	// get distance between a point (x1,y1) and landmark
	inline double getDistance(const double& x1, const double& y1, const LandmarkObs& landmark) {
		return sqrt(pow(x1 - landmark.x, 2) + pow(y1 - landmark.y, 2));
	}

	// get distance between two landmarks
	inline double getDistance(const LandmarkObs& l1, const LandmarkObs& l2) {
		return sqrt(pow(l1.x - l2.x, 2) + pow(l1.y - l2.y, 2));
	}


};


#endif /* PARTICLE_FILTER_H_ */
