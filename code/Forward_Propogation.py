#Forward Propogation

# Calculate node 0 value: node_0_value (value of input layers is defined as input data & (node_0 and node_1) are the weight of its corresponding input data) 
node_0_value = (input_data*weights["node_0"]).sum()

# Calculate node 1 value: node_1_value
node_1_value =(input_data*weights["node_1"]).sum()

# Put node values into array: hidden_layer_outputs(getting the hidden layer output and input of output layer)
hidden_layer_outputs = np.array([node_0_value, node_1_value])

# Calculate output: output
output = (hidden_layer_outputs*weights["output"]).sum()

# Print output
print(output)
