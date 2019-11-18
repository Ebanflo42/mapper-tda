import pandas as pd
import numpy as np
import abc

try:
    import params
except ImportError:
    import params_default as params
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
from mapper_help import *

class Mapper:
    """
    Implementation of the Mapper class, TDA techniques.
    """

    def __init__(self, data, Clustering, FilterFunction, overlap):

        self.N = params.N

        self.overlap = overlap

        self.data = data
        self.indices = np.arange(len(data))

        self.filter_class = FilterFunction
        self.cluster_class = Clustering

        self.filtered_values = None
        self.clusters = None
        self.cluster_graph = None
        self.centroid_graph = None

        self._implem_check()

        self.run_mapper()

    def _implem_check(self):

        if not issubclass(self.cluster_class, ClusteringTDA):
            raise TypeError('Clustering class must implement \'ClusteringTDA\' abstract class.')

        if not issubclass(self.filter_class, FilterFuctionTDA):
            raise TypeError('Filter Function class must implement \'FilterFunctionTDA\' abstract class.')

    def _apply_filter_function(self):
        fm = []
        filter_obj = self.filter_class(self.data)

        for i in self.indices:
            fm.append(filter_obj.filter_func(self.data[i:i+1], self.data))

        return pd.Series(fm, index=self.indices).sort_values()

    def _bin_data(self):
        """
         Bin filter function array into N bins with percent overlap p
         Return filter function bin membership and bin edges
        """

        finish = self.filtered_values.iloc[-1]
        start = self.filtered_values.iloc[0]

        # Size of bins, bin overlap size, bins
        bin_len = (finish-start)/self.N
        bin_over = self.overlap*bin_len
        bins = [(start + (bin_len-bin_over)*i, start + bin_len*(i+1)) for i in range(self.N)]

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

        """
        if params.CLUSTERING_PLOT_BOOL:
            for i, c in enumerate(clusters):
                c.make_plot(plot_name=params.PLOT_PATH+'hist_%s.png'%(i))
        """

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

        print("Applying Filter Function...")
        print("--------------------------------")

        # Store filter function results array
        self.filtered_values = self._apply_filter_function()

        print("Start Partial Clustering...")
        print("--------------------------------")

        # Apply clustering to each of the bins
        partial_clusters = self._apply_clustering()

        print("Building Graph...")
        print("--------------------------------")

        # Build edges between clusters of different bins if they share points
        self._build_graph(partial_clusters)

class ClusteringTDA(abc.ABC):

    """
    Abstract Clustering class to be implemented for Mapper
    """

    def __init__(self, data):
        pass

    @abc.abstractmethod
    def run_clustering(self):
        pass

    @abc.abstractmethod
    def make_plot(self, plot_name):
        pass

class FilterFuctionTDA(abc.ABC):

    """
    Abstract Filter Function class to be implemented for Mapper
    """

    def __init__(self, data):
        pass

    @abc.abstractmethod
    def filter_func(self, *args):
        pass

if __name__ == '__main__':
    pass
