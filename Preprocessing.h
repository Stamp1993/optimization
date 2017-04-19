#pragma once
#include<Eigen/Dense>
#include <fstream>
#include <iostream>
#include <vector>
#include <map>
#include<istream>
#include<iterator>
#include <sstream>
#include <fstream>
#include <string>
#include<set>
#include<random>
#include<Eigen/Eigenvalues>
using namespace Eigen;
using namespace std;

int randInt() {//random
	return (rand() * 100000) % 100;
}

map<int, VectorXd> vectorize(map<int, string> in) {
	map<int, VectorXd> res;
	set<string> items;
	for (auto i : in) {
		items.insert(i.second);
	}
	int size = items.size();
	map<string, int> offs;
	int off = 0;
	for (auto i : items) {
		offs[i] = off;
		off++;
	}
	int addr;
	for (auto j : in) {
		VectorXd vec = VectorXd::Zero(size)*0.0001;
		addr = offs[j.second];
		vec[addr] = 1;
		res[j.first] = vec;
	}

	return res;
}

map<int, VectorXd> featurize(map<int, string> in) {
	map<int, VectorXd> res;
	set<string> items;
	for (auto k : in) {
		string input = k.second;

		stringstream  lineStream(input);
		string        cell;

		while (getline(lineStream, cell, '|'))
		{
			items.insert(cell);

		}
	}
	int size = items.size();
	map<string, int> offs;
	int off = 0;
	for (auto k : items) {
		offs[k] = off;
		off++;
	}

	for (auto l : in) {

		VectorXd vec = VectorXd::Zero(size)*0.0001;

		string input = l.second;

		stringstream  lineStream(input);
		string        cell;

		while (getline(lineStream, cell, '|'))
		{
			vec[offs[cell]] = 1;
		}


		res[l.first] = vec;
	}

	return res;
}

map<int, VectorXd> date(map<int, string> in) {
	map<int, VectorXd> res;
	set<string> items;
	for (auto i : in) {
		string input = i.second;
		int id = i.first;
		stringstream  lineStream(input);
		string        cell;

		while (getline(lineStream, cell, ' '))
		{

			stringstream  ls(cell);
			string        c;

			VectorXd vec = VectorXd::Zero(3)*0.0001;
			int iter = 0;
			while (getline(ls, c, ':'))
			{
				vec[iter] = stod(c);
				iter++;
			}
			res[id] = vec;
			break;
		}
	}

	return res;
}

map<int, string> read_dat(string filename) {
    filename = "C:\\Users\\innopolis\\Documents\\Reinforcement_Marochko\\Opt\\" + filename;
	ifstream  data(filename);
	map<int, string> result;
	string line;
	int i = 0;
	int key = 0;
	while (getline(data, line))
	{
		stringstream  lineStream(line);
		string        cell;

		while (getline(lineStream, cell, ','))
		{
			cell = (cell.size()>1) ? cell.substr(1, cell.size() - 2) : cell;

			if (i == 0) {
				key = stoi(cell);
				i++;
			}
			else {
				stringstream  ls(cell);
				string        c;
				while (getline(ls, c, ' '))
				{
					result[key] = c;
					break;
				}
			}
				
		}
		i--;
	}
	return result;
}

map<int, int> read_dat_int(string filename) {
    filename = "C:\\Users\\innopolis\\Documents\\Reinforcement_Marochko\\Opt\\" + filename;
	ifstream  data(filename);
	map<int, int> result;
	string line;
	int i = 0;
	int key = 0;
	while (getline(data, line))
	{
		stringstream  lineStream(line);
		string        cell;

		while (getline(lineStream, cell, ','))
		{
			cell = (cell.size()>1) ? cell.substr(1, cell.size() - 2) : cell;

			if (i == 0) {
				key = stoi(cell);
				i++;
			}
			else
				result[key] = stoi(cell);
		}
		i--;
	}
	return result;
}

map<int, double> read_dat_dbl(string filename) {
    filename = "C:\\Users\\innopolis\\Documents\\Reinforcement_Marochko\\Opt\\" + filename;
	ifstream  data(filename);
	map<int, double> result;
	string line;
	int i = 0;
	int key = 0;
	while (getline(data, line))
	{
		stringstream  lineStream(line);
		string        cell;

		while (getline(lineStream, cell, ','))
		{
			cell = (cell.size()>1) ? cell.substr(1, cell.size() - 2) : cell;

			if (i == 0) {
				key = stoi(cell);
				i++;
			}
			else
				result[key] = stod(cell);
		}
		i--;
	}
	return result;
}
