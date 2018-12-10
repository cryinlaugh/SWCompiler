/*************************************************************************
	> File Name: dotGen.h
	> Author: cryinlaugh
	> Mail: cryinlaugh@gmail.com
	> Created Time: Wed Dec  5 08:52:05 2018
 ************************************************************************/

#ifndef _DOTGEN_H
#define _DOTGEN_H

#include <fstream>
#include <iostream>
#include <stdlib.h>

#include "SWC.h"

using namespace std;
using namespace swc;

void dotGen(IRGraph<float> graph, string dotFileName) {

	cout << "Generate the dotFile for drawing." << endl;

	string dot_Total;

	for (int i = 0; i < graph.ternsorNodeNum(); i++) {
		dot_Total = dot_Total + graph.getTensorNode(i)->dotGen();
	}

	for (int i = 0; i < graph.opNodeNum(); i++) {
		dot_Total = dot_Total + graph.getOpNode(i)->dotGen();
	}

	string dot_title = "digraph CG { \n";
	string dot_end   = "\n}";

	// dotFile Genrate
	ofstream dotfile(dotFileName, fstream::out);

	dotfile << dot_title << endl;
	dotfile << dot_Total;
	dotfile << dot_end << endl;

	// make PNG
	string svgFileName = "IRGraph.svg";
	string dotGenCMD   = "dot -T svg " + dotFileName + " -o " + svgFileName;

	char *cmd = (char*)dotGenCMD.data();

	system(cmd);
}

void dotGen(IRGraph<float> graph) {

	dotGen(graph, "IRGraph.dot");

}

#endif
