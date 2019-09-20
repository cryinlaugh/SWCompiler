/*
 * IRNode.h
 * Copyright (C) 2018 Hongkun Yu <staryhk@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef IRNODE_H_
#define IRNODE_H_

#include "common.h"

#include <iostream>
#include <string>
#include <unordered_map>
#include <algorithm>

#include "pass/Label.h"
#include "pass/AutodiffPass.h"

namespace swc {

//forward declaration
class IRGraph;


class IRNode {
  public:
    IRNode();
    IRNode(const NodeType nodeType, std::string name, IRNode *parent = nullptr)
        : _name(name), _nodeType(nodeType), _label(new Label()) {
        _topologyId = 0;
        if (parent)
            exlinkUpperNode(parent);
    }
    ~IRNode() { printf("free:%s\n", _name.c_str()); }

    void pushParentNode(){};
    template <typename T, typename... Types>
    void pushParentNode(const T &firstArg, const Types &... args) {
        _parentNodes.push_back(firstArg);
        pushParentNode(args...);
    }

    void delParentNode(){};
    template <typename T, typename... Types>
    void delParentNode(const T &firstArg, const Types &... args) {
        if (!delVecMember(_parentNodes, firstArg)) {
            std::cout << "Del Parent Failed" << firstArg->name() << std::endl;
        }
        delParentNode(args...);
    }

    void pushChildNode(){};
    template <typename T, typename... Types>
    void pushChildNode(const T &firstArg, const Types &... args) {
        _childNodes.push_back(firstArg);
        pushChildNode(args...);
    }

    void delChildNode(){};
    template <typename T, typename... Types>
    void delChildNode(const T &firstArg, const Types &... args) {
        if (!delVecMember(_childNodes, firstArg)) {
            std::cout << "Del Parent Failed" << firstArg->name() << std::endl;
        }
        delChildNode(args...);
    }

    void exlinkUpperNode(){};
    template <typename T, typename... Types>
    void exlinkUpperNode(const T &firstArg, const Types &... args) {
        pushParentNode(firstArg);
        firstArg->pushChildNode(this);
        exlinkUpperNode(args...);
    }

    void destroyUpperNode(){};
    template <typename T, typename... Types>
    void destroyUpperNode(const T &firstArg, const Types &... args) {
        delParentNode(firstArg);
        firstArg->delChildNode(this);
        destroyUpperNode(args...);
    }

    const std::vector<IRNode *> &getParentNodes() const { return _parentNodes; }
    const std::vector<IRNode *> &getChildNodes() const { return _childNodes; }

    std::vector<IRNode *> &getParentNodes() { return _parentNodes; }
    std::vector<IRNode *> &getChildNodes() { return _childNodes; }

    IRNode *getParentNode(int i) const { return _parentNodes[i]; }
    IRNode *getChildNode(int i) const { return _childNodes[i]; }

    inline const std::string name() const { return _name; };
    inline void setName(std::string name) { _name = name; };

    inline int parentNum() const { return _parentNodes.size(); }
    inline int childNum() const { return _childNodes.size(); }

    inline int topologyId() const { return _topologyId; }
    inline void setTopologyId(int topologyId) { _topologyId = topologyId; }

    inline NodeType nodeType() const { return _nodeType; }
    inline void setNodeType(NodeType nodeType) { _nodeType = nodeType; }

    void replaceUseKeepOrder(IRNode *node) {
        /*
        for(auto p : _parentNodes){
            for(auto &pc : p->getChildNodes()){
                if(pc == this){
                    pc = node;
                    node->pushParentNode(p);
                }
            }
        }
        */

        for (auto c : _childNodes) {
            for (auto &cp : c->getParentNodes()) {
                if (cp == this) {
                    cp = node;
                    this->delChildNode(c);
                    node->pushChildNode(c);
                }
            }
        }
    }
    void replaceUseKeepOrder(IRNode *spec_child, IRNode *node) {
        std::cout << "[[[replaceUseKeepOrder]]]\n";
        if(std::find(_childNodes.begin(), _childNodes.end(), spec_child) == _childNodes.end())
            return;
        for(auto n : spec_child->getParentNodes())
            std::cout<< n->name() << "\n";
        std::cout << "replaceUseKeepOrder begin\n";
        for (auto &cp : spec_child->getParentNodes()) {
            if (cp == this) {
                // order of parent matter
                cp = node;
                // original link to spec_child (may have other children)
                this->delChildNode(spec_child);
                // node is new added, so order does not matter
                node->pushChildNode(spec_child);
            }
        }
        for(auto n : spec_child->getParentNodes())
            std::cout<< n->name() << "\n";
    }

    void setLabel(Label *label) { _label = label; }
    Label *getLabel() const { return _label; }

    void setExternal(bool flag) { _isExternal = flag; }
    bool isExternal() const { return _isExternal; }

    //Virtual function entry
    virtual IRNode *clone() const = 0;
    virtual IRNode *deepClone() const = 0;

    virtual void destroy(){};
    virtual void autoDiff(IRGraph* graph,
                        std::unordered_map<IRNode*, IRNode*> &gradNodeMap){};

    virtual void autoDiff(IRGraph* graph,
                        std::unordered_map<IRNode*, IRNode*> &gradNodeMap,
                        void* methodParams,
                        pass::METHOD_TYPE methodType){};

    virtual void checkValid(){};



  private:
    std::vector<IRNode *> _parentNodes;
    std::vector<IRNode *> _childNodes;
    std::string _name;

    NodeType _nodeType;
    int _topologyId;
    Label *_label;

    bool _isExternal{false};
};

} // namespace swc

#endif /* !IRNODE_H_ */
