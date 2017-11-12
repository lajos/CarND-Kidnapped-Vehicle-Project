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

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	num_particles = 1000;

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

void ParticleFilter::updateParticle(Particle& p, const double& v, const double& yaw_dot, const double& dt) {
	double yaw_new = p.theta + yaw_dot * dt;
	p.x = p.x + v / yaw_dot * (sin(yaw_new) - sin(p.theta));
	p.y = p.y + v / yaw_dot * (cos(p.theta) - cos(yaw_new));
	p.theta = constrainRadian(yaw_new);
}

void ParticleFilter::addParticleNoise(Particle& p, const double std_pos[]) {
	default_random_engine gen;

	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);

	p.x += dist_x(gen);
	p.y += dist_y(gen);
	p.theta += dist_theta(gen);
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	for (auto it = particles.begin(); it != particles.end(); ++it) {
		updateParticle(*it, velocity, yaw_rate, delta_t);
		addParticleNoise(*it, std_pos);
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

}

inline void ParticleFilter::addObservation(Particle& p, const LandmarkObs& obs, const double& sensor_range, vector<LandmarkObs>& p_observations) {
	const double yaw = constrainRadian(p.theta);
	const double cosYaw = cos(yaw);
	const double sinYaw = sin(yaw);
	double t_x = p.x + cosYaw * obs.x - sinYaw * obs.y;  // transformed x
	double t_y = p.y + sinYaw * obs.x + cosYaw * obs.y;  // transformed y
	p.sense_x.push_back(t_x);
	p.sense_y.push_back(t_y);
	LandmarkObs t_obs = { -1, t_x, t_y };
	p_observations.push_back(t_obs);
	return;
}


void ParticleFilter::findNearestLandmarks(Particle& p, const vector<LandmarkObs>& landmarks) {

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

		vector<LandmarkObs> valid_landmarks;

		for (auto m_it = map_landmarks.landmark_list.begin(); m_it != map_landmarks.landmark_list.end(); ++m_it) {
			if (getDistance(px, py, m_it->x_f, m_it->y_f) < sensor_range) {
				LandmarkObs landmark = { m_it->id_i, m_it->x_f, m_it->y_f };
				valid_landmarks.push_back(landmark);
			}
		}

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
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

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
