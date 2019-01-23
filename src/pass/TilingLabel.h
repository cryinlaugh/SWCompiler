/***********************************************
#
#      Filename: TilingLable.h
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
#include "../graphIR/TensorNode.h"
#include "Label.h"
#include "TileHint.h"
namespace swc{



//struct TileHint{

    //int Ndim;
    //unsigned long  step;
    // std::vector<unsigned long>* step;

//}





// class TensorShape{
// private:
//     int _ndim;
//     std::vector<unsigned long>* _shape;
// public:
//     TensorShape(std::vector<unsigned long>* shape);
//     ~TensorShape(){};
//     const int getNDim() const;
//     const unsigned long getDim(int idx) const;
// };

// template <typename Dtype>
// class Tensor{
// private:
//     TensorType _type;
//     TensorShape* _shape;
//     std::shared_ptr<SWMem<Dtype> > _data;

// public:
//     Tensor(){ 
//         _type = UNKNOWN;
//         _shape = NULL;
//         _data = NULL;
//     }
//     Tensor(TensorShape* shape){
//         _type = TensorType(shape->getNDim());
//         _shape = shape;
//     }
//     ~Tensor(){}; 

//     const int getNDim() const{
//         return _shape->getNDim();
//     };
//     const unsigned long getDim(int dim) const{
//         return _shape->getDim(dim);
//     };

//     TensorShape* getTensorShape() const{
//         return _shape;

//     }
// };



class TilingLabel:public Label{
private:
    Label* _label;
    std::vector<TensorShape*> _slices;
    
public:
    TilingLabel(Label *label) : _label(label){};
    ~TilingLabel(){};

    // void destroy(){
    //     this->~TilingLabel();
    // }

   std::vector<std::vector <unsigned long>> computeSpilts (TensorShape* shape, std::vector<unsigned long > tilehintStep ){
       
       //std::vector<TensorShape>* result;
       std::vector<std::vector<unsigned long >> shapeSpilts; 
       //std::vector<unsigned long>::iterator inter;
       for(int i=0;i<shape->getNDim();++i){
           unsigned long idim=shape->getDim(i);
           unsigned long step = tilehintStep[i];
           std::vector<unsigned long > dimSpilts;
           for(unsigned long j=0;j<idim;j+=step){
               dimSpilts.push_back(std::min(idim, j+step));
           }
           shapeSpilts.push_back(dimSpilts);            
       }
       //result.push_back();
       //return result;
       return shapeSpilts;
   }

   std::vector<TensorShape>* computeSlices(TensorShape * shape, std::vector<unsigned long > tilehintStep){


       std::vector<TensorShape> * slices;
       std::vector<std::vector<unsigned long >> shapeSpilts = computeSpilts(shape, tilehintStep);
       //compute catesian product
       auto n = shapeSpilts.size();
       auto next = [&](std::vector<unsigned long > &x){
           for (int i=0;i< n; ++i)
               if( ++x[i] == shapeSpilts[i].size()) x[i]=0;
               else return true;
           return false;
       };
       // auto assemble = [&](std::vector<int> const& x){
       //      std::vector<unsigned long> item;
       //      for( int i=0;i <n ;++i)

       //          //slices->push_back(shapeSpilts[i][x[i]]);
       // };
       auto print = [&](std::vector<unsigned long> const& x) {
            for ( int i = 0; i < n; ++ i ) 
                std::cout << shapeSpilts[i][x[i]] << ",";
                std::cout << "\b \n";
        };
       std::vector<unsigned long > x(n);
       do print(x); while(next(x));

       return slices;



   }
   std::vector<TensorShape*> getSlices(){
       return _slices;
   }



    void setTilingLabel(TileHint tilehint){

        TensorShape* shape=tilehint.getTensorShape();
        std::vector<unsigned long> steps=tilehint.getTilingStep();

        //std::cout<<"set tiling label"<<std::endl;
       
       //tilehint

        computeSlices(shape,steps);
    }

    //void setNodeNameLabel(std::string s) { _nodeNameLabel = s; };
    //void setTypeNameLabel(std::string s) { _typeNameLabel = s; };

    //void setLowerMark() { _toLower = 1;};

    //std::string getNodeNameLabel() const { return _nodeNameLabel; };
    //std::string getTypeNameLabel() const { return _typeNameLabel; };

    //int getLowerMark() const { return _toLower;};


};

}
#endif
