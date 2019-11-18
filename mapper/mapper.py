import pandas as pd
import numpy as np
import types


class Mapper:

    def __init__(self, data, Clustering, num_bins, filter_function, overlap):

        self.num_bins = num_bins
        self.overlap = overlap

        self.data = data
        self.indices = np.arange(len(data))

        self.filter_function = filter_function
        self.cluster_class = Clustering

        self.filtered_values = None
        self.clusters = None
        self.centroids = None
        self.graph = None

        self._check_implem()

        self.run_mapper()

    def _check_implem(self):
        if isinstance(self.filter_function, types.LambdaType):
            return
        else: raise TypeError('`filter_function` must be callable.')

    def _apply_filter_function(self):

        fm = []
        for i in self.indices:
            fm.append(self.filter_function(self.data[i]))

        return pd.Series(fm, index=self.indices).sort_values()

    def _bin_data(self):
        """
         Bin filter function array into N bins with percent overlap p
         Return filter function bin membership and bin edges
        """

        finish = self.filtered_values.iloc[-1]
        start = self.filtered_values.iloc[0]

        # Size of bins, bin overlap size, bins
        bin_len = (finish-start)/self.num_bins
        bin_over = self.overlap*bin_len
        bins = [(start + (bin_len-bin_over)*i, start + bin_len*(i+1)) for i in range(self.num_bins)]

        binned_dict = {}
        for edge in bins:
            bool_corr = self.filtered_values.apply(lambda x: True if x>=edge[0] and x<=edge[1] else False)
            binned_dict[edge] = self.filtered_values[bool_corr]

        return binned_dict, bins

    def _apply_clustering(self):
        binned_dict, bins = self._bin_data()

        partial_clusters = {}
        counter = 0
        node_colors = {}

        clusters = []

        for i, k in enumerate(bins):

            keys = list(binned_dict[k].index)

            local_to_global = dict(zip(list(range(len(self.data))), keys))

            cluster_obj = self.cluster_class(self.data[keys])

            clusters.append( cluster_obj )

            c_to_ind = cluster_obj.run_clustering()

            global_cluster_names = {}
            for c in c_to_ind:
                global_cluster_names[counter] = [local_to_global[local_index] for local_index in c_to_ind[c]]
                node_colors[counter] = np.mean(binned_dict[k])
                counter += 1

            partial_clusters[i] = global_cluster_names

        return partial_clusters

    def _build_graph(self, partial_clusters):

        G = nx.Graph()

        for k in range(len(partial_clusters)-1):
            for c in partial_clusters[k]:
                G.add_node(c)

        for k in range(len(partial_clusters)-1):
            for c1 in partial_clusters[k]:
                for c2 in partial_clusters[k+1]:
                    if set(partial_clusters[k][c1]).intersection(partial_clusters[k+1][c2]):
                        G.add_edge(c1, c2)

        self.graph = G

    def _get_centroid_graph(self, partial_clusters):

        c_to_centroid = {}
        for _, clusters in partial_clusters.items():
            for node, indices in clusters.items():
                color = node_colors[node]
                ax.scatter(*np.mean(self.data[indices], axis=0), s=200, c=color, lw=0, alpha=.4)
                c_to_centroid[node] = np.mean(self.data[indices], axis=0)

        for e in self.graph.edges:
            e1, e2 = e
            x = c_to_centroid[e1]
            y = c_to_centroid[e2]
            ax.plot(*zip(x,y), ms=0, ls='-', lw=1., color='k')

        self.centroid_graph = c_to_centroid

    def run_mapper(self):

        # Store filter function results array
        self.filtered_values = self._apply_filter_function()

        # Apply clustering to each of the bins
        partial_clusters = self._apply_clustering()

        # Build edges between clusters of different bins if they share points
        self._build_graph(partial_clusters)
