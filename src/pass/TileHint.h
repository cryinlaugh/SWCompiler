/***********************************************
#
#      Filename: TilingLable.h
#
#        Author: whl - lethewang@yeah.net
#   Description: ---
#        Create: 2019-01-21 17:54:42
# Last Modified: 2019-01-21 17:54:42
***********************************************/

#ifndef _TILEHINT_H
#define _TILEHINT_H

#include <string>
#include <vector>

namespace swc{


class TileHint{
private:

	//TensorShape* _shape;
     

    //std::vector<unsigned long> _steps;
    //int _numshards;

public:
    // TileHint(TensorShape *shape):_shape(shape){

    // 	for(int i=0;i<shape->getNDim();++i){
    // 		_steps.push_back(shape->getDim(i));
    // 	}

    // }
    // ~TileHint(){};


    // void setSimpleTilingByDim(int dim){

    // 	//simple spilt case
    // 	unsigned long step = _shape->getDim(dim)/_numshards;
    // 	_steps[dim]=step;

    // };








    // std::vector<unsigned long> getTilingStep(){

    // 	return _steps;

    // }




    // TensorShape* getTensorShape(){

    // 	return _shape;

    // }


 // std::vector<std::vector <unsigned long>> computeSpilts (TensorShape* shape, std::vector<unsigned long > tilehintStep ){
       
 //       //std::vector<TensorShape>* result;
 //       std::vector<std::vector<unsigned long >> shapeSpilts; 
 //       //std::vector<unsigned long>::iterator inter;
 //       for(int i=0;i<shape->getNDim();++i){
 //           unsigned long idim=shape->getDim(i);
 //           unsigned long step = tilehintStep[i];
 //           std::vector<unsigned long > dimSpilts;
 //           for(unsigned long j=0;j<idim;j+=step){
 //               dimSpilts.push_back(std::min(idim, j+step));
 //           }
 //           shapeSpilts.push_back(dimSpilts);            
 //       }
 //       //result.push_back();
 //       //return result;
 //       return shapeSpilts;
 //   }

 //   std::vector<TensorShape>* computeSlices(TensorShape * shape, std::vector<unsigned long > tilehintStep){


 //       std::vector<TensorShape> * slices;
 //       std::vector<std::vector<unsigned long >> shapeSpilts = computeSpilts(shape, tilehintStep);
 //       //compute catesian product
 //       auto n = shapeSpilts.size();
 //       auto next = [&](std::vector<unsigned long > &x){
 //           for (int i=0;i< n; ++i)
 //               if( ++x[i] == shapeSpilts[i].size()) x[i]=0;
 //               else return true;
 //           return false;
 //       };
 //       // auto assemble = [&](std::vector<int> const& x){
 //       //      std::vector<unsigned long> item;
 //       //      for( int i=0;i <n ;++i)

 //       //          //slices->push_back(shapeSpilts[i][x[i]]);
 //       // };
 //       auto print = [&](std::vector<unsigned long> const& x) {
 //            for ( int i = 0; i < n; ++ i ) 
 //                std::cout << shapeSpilts[i][x[i]] << ",";
 //                std::cout << "\b \n";
 //        };
 //       std::vector<unsigned long > x(n);
 //       do print(x); while(next(x));

 //       return slices;



 //   }



};

}
#endif