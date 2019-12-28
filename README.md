# mapper-tda

This is a fork of the `mapper-tda` library by Kiran Sanjeevan.

This fork allows the use of the DBSCAN clustering algorithm for `mapper`; I believe it is easier to heuristically determine the parameters for DBSCAN than it is for SLC, plus DBSCAN discards noise.

Additionally, it includes a filter function which treats the data as if it were a set of trajectories sample from some vector field, and then filters based on the distance along each trajectory. I think that this may be useful for applying topological NLDR methods to vector field data.