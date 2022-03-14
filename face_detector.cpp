#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui.hpp"

#include <iostream>

const char* keys = 
{
	"{ i input |  | The source image }"
	"{ d dir   |  | The resource directory }"
};

int main(int argc, const char** argv)
{
	cv::CommandLineParser parser(argc, argv, keys);
	std::string infile = parser.get<std::string>("input");
	std::cout << "Input: '" << infile << "'" << std::endl;

	std::string dir = parser.get<std::string>("dir");
	std::cout << "Dir: '" << dir << "'" << std::endl;

	std::string imgdir = "out";
	std::string resdir = "res";
	std::string outdir = dir + imgdir;
	std::string cascade_xml = "/haarcascade_frontalface_alt.xml";
	std::string cascade_file = dir + resdir + cascade_xml;

	std::cout << "Cascade file: " << cascade_file << std::endl;

	cv::CascadeClassifier cascade;
	if (cascade_file.empty() || !cascade.load(cascade_file))
	{
		std::cout << cv::format("Error: cannot load cascade file! %s - %s\n", outdir.c_str(), cascade_file.c_str());
		return -1;
	}

	std::cout << "Source image: " << infile << std::endl;

	cv::Mat src = cv::imread(infile);
	if (src.empty())
	{
		std::cout << cv::format("Error: cannot load source image!\n");
		return -1;
	}

	cv::Mat gray;
	cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
	cv::equalizeHist(gray, gray);

	std::vector<cv::Rect> faces;
	cascade.detectMultiScale(gray, faces, 1.2, 3);

	std::cout << cv::format("0, %s (%dx%d)\n", infile.c_str(), src.cols, src.rows);

	cv::Mat src_copy = src.clone();
	for (int i = 0; i < faces.size(); i++)
	{
		std::string outfile(cv::format("%s/face-%d.jpeg", outdir.c_str(), i+1));
		std::string outfile_marks(cv::format("%s/face-%d-marks.jpeg", outdir.c_str(), i+1));
		cv::Rect r = faces[i];
		cv::rectangle(src, r, CV_RGB(0,255,0), 2);
		cv::imwrite(outfile, src_copy(r));
		cv::imwrite(outfile_marks, src);
		std::cout << cv::format("%d, %s (%dx%d)\n", i+1, outfile.c_str(), r.width, r.height);
		std::cout << cv::format("%d, %s (%dx%d)\n", i+1, outfile_marks.c_str(), r.width, r.height);
	}

	return 0;
}