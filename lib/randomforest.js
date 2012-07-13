// MIT License
// Andrej Karpathy

var dectreejs = (function(){
  
  var RandomForest = function(options) {
  }
  RandomForest.prototype = {
  
    train: function(data, labels, options) {
    
      options = options || {};
      this.numTrees = options.numTrees || 100;
      
      // initialize many trees and train them all independently
      this.trees= new Array(this.numTrees);
      for(var i=0;i<this.numTrees;i++) {
        this.trees[i] = new DecisionTree();
        this.trees[i].train(data, labels, options);
      }
    },
    
    predictOne: function(inst) {
      
      // have each tree predict and average out all votes
      var dec=0;
      for(var i=0;i<this.numTrees;i++) {
        dec += this.trees[i].predictOne(inst);
      }
      dec /= this.numTrees;
      return dec;
    }
  }
  
  var DecisionTree = function(options) {
  }
  
  DecisionTree.prototype = {
  
    train: function(data, labels, options) {
      
      options = options || {};
      var maxDepth = options.maxDepth || 4;
      
      var trainFun= decision2DStumpTrain;
      var testFun= decision2DStumpTest;
      
      if(options.trainFun) trainFun = options.trainFun;
      if(options.testFun) testFun = options.testFun;
      
      if(options.type == 0) {
        trainFun= decisionStumpTrain;
        testFun= decisionStumpTest;
      }
      if(options.type == 1) {
        trainFun= decision2DStumpTrain;
        testFun= decision2DStumpTest;
      }
      
      // initialize helper variables
      var numInternals= Math.pow(2, maxDepth)-1;
      var numNodes= Math.pow(2, maxDepth + 1)-1;
      var ixs= new Array(numNodes);
      for(var i=1;i<ixs.length;i++) ixs[i]=[];
      ixs[0]= new Array(labels.length);
      for(var i=0;i<labels.length;i++) ixs[0][i]= i; // root node starts out with all nodes as relevant
      var models = new Array(numInternals);
      
      // train
      for(var n=0; n < numInternals; n++) {
        
        // few base cases
        var ixhere= ixs[n];
        if(ixhere.length == 0) { continue; }
        if(ixhere.length == 1) { ixs[n*2+1] = [ixhere[0]]; continue; } // arbitrary send it down left
        
        // learn a weak model on relevant data for this node
        var model= trainFun(data, labels, ixhere);
        models[n]= model; // back it up model
        
        // split the data according to the learned model
        var ixleft=[];
        var ixright=[];
        for(var i=0; i<ixhere.length;i++) {
            var label= testFun(data[ixhere[i]], model);
            if(label === 1) ixleft.push(ixhere[i]);
            else ixright.push(ixhere[i]);
        }
        ixs[n*2+1]= ixleft;
        ixs[n*2+2]= ixright;
      }
      
      // compute data distributions at the leafs
      var leafPositives = new Array(numNodes);
      var leafNegatives = new Array(numNodes);
      for(var n=numInternals; n < numNodes; n++) {
        var numones= 0;
        for(var i=0;i<ixs[n].length;i++) {
            if(labels[ixs[n][i]] === 1) numones+=1;
        }
        leafPositives[n]= numones;
        leafNegatives[n]= ixs[n].length-numones;
      }
      
      // back up important prediction variables for predicting later
      this.models= models;
      this.leafPositives = leafPositives;
      this.leafNegatives = leafNegatives;
      this.maxDepth= maxDepth;
      this.trainFun= trainFun;
      this.testFun= testFun;
    }, 
    
    predictOne: function(inst) { 
        
        var n=0;
        for(var i=0;i<this.maxDepth;i++) {
            var dir= this.testFun(inst, this.models[n]);
            if(dir === 1) n= n*2+1; // descend left
            else n= n*2+2; // descend right
        }
        
        return (this.leafPositives[n] + 0.5) / (this.leafNegatives[n] + 1.0); // bayesian smoothing!
    }
  }
  
  // returns model
  function decisionStumpTrain(data, labels, ix, options) {
    
    options = options || {};
    var numtries = options.numTries || 10;
    
    // choose a dimension at random and pick a best split
    var ri= randi(0, data[0].length);
    var N= ix.length;
    
    // evaluate class entropy of incoming data
    var H= entropy(labels, ix);
    var bestGain=0; 
    var bestThr= 0;
    for(var i=0;i<numtries;i++) {
    
        // pick a random splitting threshold
        var ix1= ix[randi(0, N)];
        var ix2= ix[randi(0, N)];
        var a= Math.random()*1.5-0.75;
        var thr= data[ix1][ri]*a + data[ix2][ri]*(1-a);
        
        // measure information gain we'd get from split with thr
        var l1=1, r1=1, lm1=1, rm1=1; //counts for Left and label 1, right and label 1, left and minus 1, right and minus 1
        for(var j=0;j<ix.length;j++) {
            if(data[ix[j]][ri] < thr) {
              if(labels[ix[j]]==1) l1++;
              else lm1++;
            } else {
              if(labels[ix[j]]==1) r1++;
              else rm1++;
            }
        }
        var t= l1+lm1; 
        l1=l1/t;
        lm1=lm1/t;
        t= r1+rm1;
        r1=r1/t;
        rm1= rm1/t;
        
        var LH= -l1*Math.log(l1) -lm1*Math.log(lm1); // left and right entropy
        var RH= -r1*Math.log(r1) -rm1*Math.log(rm1);
        
        var informationGain= H - LH - RH;
        //console.log("Considering split %f, entropy %f -> %f, %f. Gain %f", thr, H, LH, RH, informationGain);
        if(informationGain > bestGain || i === 0) {
            bestGain= informationGain;
            bestThr= thr;
        }
    }
    
    model= {};
    model.thr= bestThr;
    model.ri= ri;
    return model;
  }
  
  // returns label for a single data instance
  function decisionStumpTest(inst, model) {
    if(!model) {
        // this is a leaf that never received any data... 
        return 1;
    }
    return inst[model.ri] < model.thr ? 1 : -1;
    
  }
  
  // returns model
  function decision2DStumpTrain(data, labels, ix, options) {
    
    options = options || {};
    var numtries = options.numTries || 10;
    
    // choose a dimension at random and pick a best split
    var N= ix.length;
    
    // evaluate class entropy of incoming data
    var H= entropy(labels, ix);
    var bestGain=0; 
    var bestw1, bestw2, bestb;
    for(var i=0;i<numtries;i++) {
    
        // pick random line parameters
        var w1= randf(-2, 2);
        var w2= randf(-2, 2);
        var b= randf(-3, 3);
        
        // measure information gain we'd get from split with thr
        var l1=1, r1=1, lm1=1, rm1=1; //counts for Left and label 1, right and label 1, left and minus 1, right and minus 1
        for(var j=0;j<ix.length;j++) {
            var pp= data[ix[j]][0]*w1 + data[ix[j]][1]*w2 + b;
            if(pp < 0) {
              if(labels[ix[j]]==1) l1++;
              else lm1++;
            } else {
              if(labels[ix[j]]==1) r1++;
              else rm1++;
            }
        }
        var t= l1+lm1; 
        l1=l1/t;
        lm1=lm1/t;
        t= r1+rm1;
        r1=r1/t;
        rm1= rm1/t;
        
        var LH= -l1*Math.log(l1) -lm1*Math.log(lm1); // left and right entropy
        var RH= -r1*Math.log(r1) -rm1*Math.log(rm1);
        
        var informationGain= H - LH - RH;
        //console.log("Considering split %f, entropy %f -> %f, %f. Gain %f", thr, H, LH, RH, informationGain);
        if(informationGain > bestGain || i === 0) {
            bestGain= informationGain;
            bestw1= w1;
            bestw2= w2;
            bestb= b;
        }
    }
    
    model= {};
    model.w1= bestw1;
    model.w2= bestw2;
    model.b= bestb;
    return model;
  }
  
  // returns label for a single data instance
  function decision2DStumpTest(inst, model) {
    if(!model) {
        // this is a leaf that never received any data... 
        return 1;
    }
    return inst[0]*model.w1 + inst[1]*model.w2 + model.b < 0 ? 1 : -1;
    
  }
  
  // Misc utility functions
  function entropy(labels, ix) {
    var N= ix.length;
    var p=0.0;
    for(var i=0;i<N;i++) {
        if(labels[ix[i]]==1) p+=1;
    }
    p=(1+p)/(N+2);
    q=(1+N-p)/(N+2);
    return (-p*Math.log(p) -q*Math.log(q));
  }
  
  // generate random floating point number between a and b
  function randf(a, b) {
    return Math.random()*(b-a)+a;
  }

  // generate random integer between a and b (b excluded)
  function randi(a, b) {
     return Math.floor(Math.random()*(b-a)+a);
  }

  // export public members
  var exports = {};
  exports.DecisionTree = DecisionTree;
  exports.RandomForest = RandomForest;
  return exports;
  
})();
