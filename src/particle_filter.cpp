/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <map> // TODO remove

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	num_particles = 25;

	double init_weight = 1.0 / num_particles;

	default_random_engine gen;

	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	for (size_t i = 0; i < num_particles; ++i) {
		Particle p;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = init_weight;
		particles.push_back(p);
	}

	is_initialized = true;
}


void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	default_random_engine gen;

	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);

	for (auto it = particles.begin(); it != particles.end(); ++it) {

		// update particle position and yaw using bicycle model
		//
		double yaw = it->theta;
		double yaw_new = yaw + yaw_rate * delta_t;
		if (yaw_rate < 0.0000001) {			// avoid division by zero if yaw change too small
			it->x += delta_t *velocity * cos(yaw_new);
			it->y += delta_t *velocity * sin(yaw_new);
		} else {
			it->x += velocity / yaw_rate * (sin(yaw_new) - sin(yaw));
			it->y += velocity / yaw_rate * (cos(yaw) - cos(yaw_new));
		}
		it->theta = yaw_new;

		// add noise
		//
		it->x += dist_x(gen);
		it->y += dist_y(gen);
		it->theta = constrainRadian(it->theta + dist_theta(gen));
	}
}

inline void ParticleFilter::addObservation(Particle& p, const LandmarkObs& obs, const double& sensor_range, vector<LandmarkObs>& p_observations) {
	const double yaw = p.theta;
	const double cosYaw = cos(yaw);
	const double sinYaw = sin(yaw);
	double t_x = p.x + cosYaw * obs.x - sinYaw * obs.y;  // transformed x
	double t_y = p.y + sinYaw * obs.x + cosYaw * obs.y;  // transformed y
	LandmarkObs t_obs = { -1, t_x, t_y };
	p_observations.push_back(t_obs);
	return;
}

double ParticleFilter::calculateParticleWeight(const vector<LandmarkObs>& pObservations, const vector<LandmarkObs>& pLandmarks, const double sigma[]) {
	auto o = pObservations.begin();
	auto l = pLandmarks.begin();

	double weight = 1;

	double temp1 = 1.0 / (2 * M_PI * sigma[0] * sigma[1]);
	double sigma2square[] = { 2 * pow(sigma[0], 2), 2 * pow(sigma[1], 2) };

	do {
		double w = temp1 * exp(-((pow(o->x - l->x, 2) / sigma2square[0]) + (pow(o->y - l->y, 2) / sigma2square[1])));
		weight *= w;
		++o;
		++l;
	} while (o != pObservations.end() && l != pLandmarks.end());

	return weight;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
								   const std::vector<LandmarkObs>& observations, const Map& map_landmarks) {

	for (auto p_it = particles.begin(); p_it != particles.end(); ++p_it) {
		vector<LandmarkObs> particle_observations;

		for (auto o_it = observations.begin(); o_it != observations.end(); ++o_it) {
			addObservation(*p_it, *o_it, sensor_range, particle_observations);
		}

		double px = p_it->x;
		double py = p_it->y;

		// create vector of landmarks in sensor range
		//
		vector<LandmarkObs> valid_landmarks;
		for (auto m_it = map_landmarks.landmark_list.begin(); m_it != map_landmarks.landmark_list.end(); ++m_it) {
			if (getDistance(px, py, m_it->x_f, m_it->y_f) < sensor_range) {
				LandmarkObs landmark = { m_it->id_i, m_it->x_f, m_it->y_f };
				valid_landmarks.push_back(landmark);
			}
		}

		// find nearest landmark to each observation
		//
		vector<LandmarkObs> nearest_landmarks;
		for (auto o_it = particle_observations.begin(); o_it != particle_observations.end(); ++o_it) {
			double best_distance = sensor_range;
			int best_index = -1;
			LandmarkObs best_landmark;
			for (auto l_it = valid_landmarks.begin(); l_it != valid_landmarks.end(); ++l_it) {
				double distance = getDistance(*o_it, *l_it);
				if (distance < best_distance) {
					best_distance = distance;
					best_index = l_it->id;
					best_landmark = *l_it;
				}
			}
			nearest_landmarks.push_back(best_landmark);
		}

		p_it->weight = calculateParticleWeight(particle_observations, nearest_landmarks, std_landmark);

		// fill out particle associations and sense parameters
		//
		p_it->associations.clear();
		p_it->sense_x.clear();
		p_it->sense_y.clear();
		for (auto l_it = nearest_landmarks.begin(); l_it != nearest_landmarks.end(); ++l_it) {
			p_it->associations.push_back(l_it->id);
			p_it->sense_x.push_back(l_it->x);
			p_it->sense_y.push_back(l_it->y);
		}
	}
}

void ParticleFilter::resample() {

	// create weights vector for discrete distribution initialization
	//
	vector<double> weights;
	double weight_sum = 0;
	for (auto p_it = particles.begin(); p_it != particles.end(); ++p_it) {
		double w = p_it->weight;
		weight_sum += w;
		weights.push_back(w);

	}

	if (weight_sum == 0) return;

	std::random_device rd;
	std::mt19937 gen(rd());
	std::discrete_distribution<size_t> d(weights.begin(), weights.end());

	// resample particles based on weight discrete distribution
	//
	vector<Particle> resampled_particles;
	for (size_t i = 0; i < num_particles; ++i) {
		resampled_particles.push_back(particles[d(gen)]);
	}

	particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y) {
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations = associations;
	particle.sense_x = sense_x;
	particle.sense_y = sense_y;

	return particle;
}

string ParticleFilter::getAssociations(Particle best) {
	vector<int> v = best.associations;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1); // get rid of the trailing space
	return s;
}

string ParticleFilter::getSenseX(Particle best) {
	vector<double> v = best.sense_x;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1); // get rid of the trailing space
	return s;
}

string ParticleFilter::getSenseY(Particle best) {
	vector<double> v = best.sense_y;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1); // get rid of the trailing space
	return s;
}
