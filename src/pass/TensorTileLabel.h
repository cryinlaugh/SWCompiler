/***********************************************
#
#      Filename: TileLable.h
#
#        Author: whl - lethewang@yeah.net
#   Description: ---
#        Create: 2019-01-21 17:54:42
# Last Modified: 2019-01-21 17:54:42
***********************************************/

#ifndef _TILELABEL_H
#define _TILELABEL_H

#include <string>
#include <vector>
#include <algorithm>
#include "TensorNode.h"
#include "Lable.h"
namespace swc{


class TilingHint{
private:
    std::vector<unsigned long> steps;
public:
    TileHint();
    ~TileHint();
}
//struct TileHint{

    //int Ndim;
    //unsigned long  step;
    // std::vector<unsigned long>* step;

//}
class TilingLabel:public Label{
private:
    std::vector<TensorShape*> _slices;
    
public:
    TilingLabel() : _toLower(0){};
    ~TilingLabel(){};

    void destroy(){
        this->~TilingLabel();
    };

//    std::vector<<vector unsigned long>> computeSpilts (TensorShape* shape, std::vector<unsigned long > tilehintStep ){
//        
//        //std::vector<TensorShape>* result;
//        std::vector<vector<unsigned long >> shapeSpilts; 
//        //std::vector<unsigned long>::iterator inter;
//        for(int i=0;i<shape->getNDim();++i){
//            unsigned long idim=getDim(i);
//            step = tilehintStep[i];
//            std::vector<unsigned long > dimSpilts;
//            for(unsigned long j=0;j<idim;j+=step){
//                dimSpilts.push_back(min(idim, j+step));
//            }
//            shapeSpilts.push_back(dimSpilts);            
//        }
//        //result.push_back();
//        //return result;
//        return shapeSpilts;
//    };
//
//    std::vector<TensorShape>* computeSlices(TensorShape * shape, std::vector<unsigned long > tilehintStep){
//
//
//        std::vector<TensorShape> * slices;
//        std::vector<vector<unsigned long >> shapeSpilts = computeSpilts(shape, tilehintStep);
//        //compute catesian product
//        auto n = shapeSpilts.size();
//        auto next = [&](std::vector<unsigned long > &x){
//            for (int i=0;i< n; ++i)
//                if( ++x[i] == items[i].size()) x[i]=0;
//                else return true;
//            return false;
//        };
//        auto assemble = [&](std::vector<int> const& x){
//            for( int i=0;i <n ;++i)
//                 slices->push_back(items[i][x[i]]);
//        }
//        std::vector<unsigned long > x(n);
//        do assemble(x); while(next(x));
//
//        return slices;
//
//
//
//    };
//    std::vector<TensorShape*> getSlices(){
//        return _slices;
//    };
//


    void setTilingLabel(){
       
//        //tilehint
            //_slices =  computeSlices();
    };
    //void setNodeNameLabel(std::string s) { _nodeNameLabel = s; };
    //void setTypeNameLabel(std::string s) { _typeNameLabel = s; };

    //void setLowerMark() { _toLower = 1;};

    //std::string getNodeNameLabel() const { return _nodeNameLabel; };
    //std::string getTypeNameLabel() const { return _typeNameLabel; };

    //int getLowerMark() const { return _toLower;};


};

}
#endif
