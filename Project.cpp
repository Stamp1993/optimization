#include"OptimizeSq.h"

int main() {
	//for future
	//map<int, VectorXd> building_id = vectorize(read_dat("building_id.csv"));
	//map<int, VectorXd> description = vectorize(read_dat("description.csv"));
	//map<int, VectorXd> display_address = vectorize(read_dat("display_address.csv"));
	//map<int, VectorXd> listing_id = vectorize(read_dat("listing_id.csv"));
	
	// read data
	map<int, double> bathrooms = (read_dat_dbl("bathrooms.csv"));
	map<int, double> bedrooms = (read_dat_dbl("bedrooms.csv"));
	map<int, VectorXd> created = date(read_dat("created.csv"));
	map<int, VectorXd> features = featurize(read_dat("features.csv"));
	map<int, double> latitude = read_dat_dbl("latitude.csv");
	map<int, double> longitude = read_dat_dbl("longitude.csv");
	map<int, VectorXd> street_address = vectorize(read_dat("street_address.csv"));
	map<int, double> price = read_dat_dbl("price.csv");
	//map<int, VectorXd> manager_id = vectorize(read_dat("manager_id.csv"));

	map<int, VectorXd> interest_level = vectorize(read_dat("interest_level.csv"));
	//creat data matrices
	int ftrsize = features[4].size();
	int stsize = street_address[4].size();
	int size = created[4].size() + ftrsize + 5 + street_address[4].size()/*+manager_id[4].size()*/;
	int rows = price.size()/5;
	MatrixXd featureMat = MatrixXd(rows, size);
	MatrixXd answers = MatrixXd(rows, 3);
	MatrixXd test = MatrixXd(rows, size);
	MatrixXd testanswers = MatrixXd(rows, 3);
	int iter = 0;
	int titer = 0;
	int chanse = 20;
	for (auto first : bathrooms) {
		
		int id = first.first;
		VectorXd one = VectorXd(1);
		one[0] = first.second/5;
		VectorXd two = VectorXd(1);
		two[0] = bedrooms[id]/5;
		VectorXd three = created[id];
		//created[id].resize(0, 0);
		VectorXd four;
		if (features.find(id) == features.end()) {
			four = VectorXd::Zero(ftrsize);
		}
		else {
			four = features[id];
			//features[id].resize(0, 0);
		}
		VectorXd five = VectorXd(1);
		five[0] = latitude[id]/360;
		VectorXd six = VectorXd(1);
		six[0] = longitude[id]/360;
		VectorXd seven(stsize);
		if (street_address.find(id) == street_address.end() || street_address[id].size()!=stsize) {
			seven = VectorXd::Zero(stsize);
		}
		else {
			seven = street_address[id];
			
		}
		VectorXd eight = VectorXd(1);
		eight[0] = price[id]/10000;
		
		

		if (randInt() < chanse) {
			if (iter >= featureMat.rows()) {
				continue;
			}
			else {
				VectorXd vec = VectorXd(size);
				assert(one.size() + two.size() + three.size() + four.size() + five.size() + six.size() + seven.size() + eight.size() == size);
				vec << one, two, three, four, five, six, seven, eight/*, nine*/;
				featureMat.row(iter) = vec;
				VectorXd ans(3);
				ans = interest_level[id];
				answers.row(iter) = ans;
				iter++;
			}
		}
		if (randInt() < chanse) {
			if (iter >= test.rows()) {
				continue;
			}
			else {
				VectorXd vec = VectorXd(size);
				vec << one, two, three, four, five, six, seven, eight/*, nine*/;
				test.row(titer) = vec;
				VectorXd ans(3);
				ans = interest_level[id];
				testanswers.row(titer) = ans;
				titer++;
			}
		}
	
	}
	//free memory
	bathrooms.clear();
	bedrooms.clear();
	created.clear();
	features.clear();
	latitude.clear();
	longitude.clear();
	street_address.clear();
	price.clear();
	//manager_id.clear();
	//train GD
	MatrixXd vars = OptimizeGD(featureMat, answers);
	//test GD
	testGD(test, testanswers, vars);

    //train Nesterog AG
	MatrixXd Nes = Nesterov(featureMat, answers);
	//test Nesterov AG
	testGD(test, testanswers, Nes);

	return 0;

}