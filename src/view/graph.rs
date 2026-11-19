use crate::AnyModule;
use crate::sequential::Sequential;
use core::fmt;
use petgraph::prelude::*;
use std::fmt::Debug;
mod dot;

#[derive(Clone, Debug)]
pub struct Perceptron {
    id: usize,
}

impl Perceptron {
    pub fn new(id: usize) -> Self {
        Self { id }
    }
}

impl fmt::Display for Perceptron {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "")
    }
}

/*
 * Note:
 * All nodes are added to the graph in increasing order.
 * Ex: [h0, h1, h2, h3, ... ]
 *      0   1   2   3   ...
 * This is important becasue it allows us to find a node's index in O(1) time.
 * Read NetGraph's "add" method to verify
 * */

#[derive(Clone, Debug)]
pub struct NetGraph {
    graph: DiGraph<Perceptron, f32>,
    model: Sequential,
    input_data: Vec<f32>,
    indices: Vec<NodeIndex>,
}

impl NetGraph {
    pub fn new(model: Sequential, input_data: &[f32]) -> Self {
        Self {
            graph: DiGraph::new(),
            model: model.clone(),
            input_data: input_data.to_vec(),
            indices: Vec::new(),
        }
    }

    fn check_input_size(model: &Sequential, input: &[f32]) {
        assert_eq!(
            model.modules[0].as_linear().in_sz,
            input.len(),
            "Input size should be unique"
        );
    }

    fn create_node(g: &mut DiGraph<Perceptron, f32>, id: usize) -> NodeIndex {
        // FIX:
        // Check why this line bellow is being called every time
        // assert!(id >= 1, "Node's ID can't be negative!");
        let p = Perceptron::new(id);
        g.add_node(p)
    }

    fn create_edge(
        g: &mut DiGraph<Perceptron, f32>,
        from: NodeIndex,
        to: NodeIndex,
        weight: f32,
    ) -> EdgeIndex {
        assert!(from < to, "This should be a digraph");
        g.add_edge(from, to, weight)
    }

    fn extract_topology(&mut self, input_size: usize) {
        self.graph.clear();

        let mut previous_layer: Vec<NodeIndex> = (0..input_size)
            .map(|i| self.graph.add_node(Perceptron::new(i)))
            .collect();

        for module in &self.model.modules {
            match module {
                AnyModule::Linear(linear) => {
                    let weights = linear.weights.storage_ref();
                    let in_features = previous_layer.len();
                    let out_features = weights.len() / in_features;

                    assert_eq!(weights.len(), in_features * out_features);

                    let current_layer: Vec<NodeIndex> = (0..out_features)
                        .map(|i| self.graph.add_node(Perceptron::new(i)))
                        .collect();

                    for out_idx in 0..out_features {
                        for in_idx in 0..in_features {
                            let weight_idx = out_idx * in_features + in_idx;

                            self.graph.add_edge(
                                previous_layer[in_idx],
                                current_layer[out_idx],
                                weights[weight_idx],
                            );
                        }
                    }
                    previous_layer = current_layer;
                }
                AnyModule::Activation(_) => {}
            }
        }
    }

    pub fn network_to_graph(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        Self::check_input_size(&self.model, &self.input_data);

        let source: &str = "src/view/artifacts/graph.dot";
        let destiny: &str = "src/view/artifacts/network.png";

        for i in 0..self.input_data.len() {
            Self::create_node(&mut self.graph, i);
        }

        let number_neurons = self.input_data.len();

        Self::extract_topology(self, number_neurons);
        Self::export_topology(self, source, destiny)?;
        Ok(())
    }

    pub fn add_node(&mut self, p: Perceptron) {
        let idx = self.graph.add_node(p);
        self.indices.push(idx);
    }

    pub fn add_edge(&mut self, from: usize, to: usize, w: f32) {
        let from_idx = self.indices[from];
        let to_idx = self.indices[to];
        self.graph.add_edge(from_idx, to_idx, w);
    }

    pub fn plot(&self) {
        println!("{:?}", self.graph);
    }
}
