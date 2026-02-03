from .rl import * # from .rl import * in production
from .lib import Lib # from .lib import Lib in production
import numpy as np
from numpy.linalg import matrix_power
from dataframe import DataFrame
import pandas as pd
import contextily as cx
from matplotlib import pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import xarray as xr
from tqdm.auto import tqdm 

class GIS:
    """
    GIS class
    """
    def __init__(self):
        self.data_layers = {}
        self.fig, self.ax = plt.subplots(figsize=(17,17))

    def add_data_layer(self, layer_name, data_path, data_type='sf', lon_column_name=None, lat_column_name=None, crs='4326'):
        if data_type == 'gdf':
            self.data_layers[layer_name] = data_path
        if data_type == 'lon_lat_csv':
            data = DataFrame(data_path)
            geo_data = GIS.lon_lat_dataframe_to_geopandas(data.get_dataframe(), lon_column_name, lat_column_name, crs=crs)
            self.data_layers[layer_name] = geo_data
        elif data_type == 'shp':
            self.data_layers[layer_name] = gpd.read_file(data_path)
        elif data_type == 'parquet':
            self.data_layers[layer_name] = gpd.read_parquet(data_path)
        elif data_type == 'geojson':
            self.data_layers[layer_name] = gpd.read_file(data_path)
        
    
    def merge_data_layers(self,
                          source_layers: list[str],
                          dest_layer_name: str,
                          ignore_missing: bool = True,
                          crs: str | None = "auto") -> gpd.GeoDataFrame:
        """
        Merge multiple existing GeoDataFrame layers into one new layer.

        Parameters
        ----------
        source_layers : list[str]
            Layer names to merge in order.
        dest_layer_name : str
            Name of the resulting merged layer.
        ignore_missing : bool
            If True skip layer names not found; else raise.
        crs : 'auto' | EPSG code | None
            Target CRS. 'auto' -> use first valid layer CRS. None -> keep each CRS (must already match).

        Returns
        -------
        GeoDataFrame
            Merged GeoDataFrame stored at self.data_layers[dest_layer_name].
        """
        if not source_layers:
            raise ValueError("source_layers list is empty")

        collected = []
        target_crs = None

        # Collect and define target CRS
        for name in source_layers:
            gdf = self.data_layers.get(name)
            if gdf is None:
                if ignore_missing:
                    continue
                else:
                    raise KeyError(f"Layer '{name}' not found")
            if target_crs is None and crs == "auto":
                target_crs = gdf.crs
            collected.append((name, gdf))

        if not collected:
            raise ValueError("No layers collected (all missing or empty list)")

        # If explicit CRS provided (e.g. 'EPSG:4326')
        if crs not in (None, "auto"):
            target_crs = gpd.CRS.from_user_input(crs)

        # Reproject & unify columns
        unified_cols = set()
        prepared = []
        for name, gdf in collected:
            # Reproject if target CRS decided
            if target_crs and gdf.crs != target_crs:
                try:
                    gdf = gdf.to_crs(target_crs)
                except Exception as e:
                    raise RuntimeError(f"Failed to reproject layer '{name}': {e}")
            # Track all columns
            unified_cols.update(gdf.columns)
            prepared.append(gdf)

        # Ensure geometry column stays; fill missing columns
        unified_cols_list = list(unified_cols)
        final_frames = []
        for gdf in prepared:
            missing = [c for c in unified_cols_list if c not in gdf.columns]
            for m in missing:
                if m == "geometry":
                    continue
                gdf[m] = pd.Series([np.nan] * len(gdf), index=gdf.index)
            # Reorder (keep geometry at end if originally at end)
            if "geometry" in gdf.columns:
                cols = [c for c in unified_cols_list if c != "geometry"] + ["geometry"]
                gdf = gdf[cols]
            else:
                gdf = gdf[unified_cols_list]
            final_frames.append(gdf)

        merged = pd.concat(final_frames, ignore_index=True)
        merged = gpd.GeoDataFrame(merged, geometry="geometry", crs=target_crs)

        self.data_layers[dest_layer_name] = merged
        print(f"Merged {len(final_frames)} layer(s) into '{dest_layer_name}' rows={len(merged)}")
        return merged
    
    
    
    def drop_close_geometries(self,
                              layer_name: str,
                              distance_m: float = 10.0,
                              in_place: bool = True,
                              use_centroid: bool = False,
                              show_progress: bool = True):
        """
        Remove rows whose geometries are closer than distance_m (meters).
        Any feature participating in a close pair is dropped.
        Prints summary statistics (removed count, percentage).
        """
        if layer_name not in self.data_layers:
            raise ValueError(f"Layer '{layer_name}' not found.")

        gdf = self.data_layers[layer_name]
        orig_count = len(gdf)
        if gdf.empty:
            print(f"[Close geometry stats] distance<={distance_m}m: dataset empty (0 removed).")
            return gdf, []

        # Decide if CRS is geographic; if so project to EPSG:3857 for meter distances
        try:
            is_geographic = gdf.crs is None or gdf.crs.is_geographic
        except Exception:
            is_geographic = True

        gdf_m = gdf.to_crs(3857) if is_geographic else gdf

        geoms = gdf_m.geometry if not use_centroid else gdf_m.geometry.centroid

        # Spatial index
        try:
            sindex = geoms.sindex
        except Exception:
            sindex = None

        to_remove = set()
        from tqdm.auto import tqdm
        for i, geom in tqdm(enumerate(geoms),
                            total=len(geoms),
                            desc="Check proximity",
                            unit="feat",
                            disable=not show_progress):
            if geom is None or geom.is_empty or i in to_remove:
                continue

            if sindex:
                minx, miny, maxx, maxy = geom.bounds
                hits = list(sindex.intersection((minx - distance_m,
                                                 miny - distance_m,
                                                 maxx + distance_m,
                                                 maxy + distance_m)))
            else:
                hits = range(len(geoms))

            for j in hits:
                if j <= i or j in to_remove:
                    continue
                other = geoms.iloc[j]
                if other is None or other.is_empty:
                    continue
                if geom.distance(other) < distance_m:
                    to_remove.add(i)
                    to_remove.add(j)

        removed_count = len(to_remove)
        pct_removed = (removed_count / orig_count * 100.0) if orig_count else 0.0

        if not to_remove:
            print(f"[Close geometry stats] distance<={distance_m}m: removed 0 / {orig_count} (0.00%).")
            if in_place:
                self.data_layers[layer_name] = gdf
            return gdf, []

        filtered = gdf.drop(index=list(to_remove))
        remaining = len(filtered)
        print(f"[Close geometry stats] distance<={distance_m}m: removed {removed_count} / {orig_count} "
              f"({pct_removed:.2f}%). Remaining: {remaining}")

        if in_place:
            self.data_layers[layer_name] = filtered

        return filtered, sorted(to_remove)
    
    
    def drop_geometries_within_same_pixel(self,
                        layer_name: str,
                        tif_path: str,
                        drop: str = "both",   # 'both' | 'first' | 'last'
                        in_place: bool = True,
                        show_progress: bool = True):
        """
        Remove geometries that fall into the same raster pixel (with tqdm progress).
        - Point: use point coords; otherwise centroid.
        - Map to (row,col); groups with size >1 trigger removal by rule.
        Returns (modified_gdf, removed_index_list).
        """
        if layer_name not in self.data_layers:
            raise ValueError(f"Layer '{layer_name}' not found.")
        if drop not in ("both", "first", "last"):
            raise ValueError("drop must be one of: 'both','first','last'.")

        import os
        import rasterio
        from rasterio.transform import Affine
        from math import floor

        gdf = self.data_layers[layer_name]
        if not os.path.exists(tif_path):
            raise FileNotFoundError(f"Raster not found: {tif_path}")

        with rasterio.open(tif_path) as ds:
            r_crs = ds.crs
            transform: Affine = ds.transform

        # Reproject to raster CRS if needed
        try:
            if gdf.crs is not None and r_crs is not None and gdf.crs != r_crs:
                gdf_work = gdf.to_crs(r_crs)
            else:
                gdf_work = gdf.copy()
        except Exception:
            gdf_work = gdf.copy()

        # Extract representative coords with progress
        xs, ys = [], []
        geom_iter = gdf_work.geometry
        for geom in tqdm(geom_iter, desc="Map geometries", unit="feat", disable=not show_progress):
            if geom is None or geom.is_empty:
                xs.append(np.nan); ys.append(np.nan)
            elif geom.geom_type == "Point":
                xs.append(geom.x); ys.append(geom.y)
            else:
                c = geom.centroid
                xs.append(c.x); ys.append(c.y)

        # Pixel indices
        inv = ~transform
        rows, cols = [], []
        for x, y in tqdm(zip(xs, ys),
                         total=len(xs),
                         desc="Compute pixel indices",
                         unit="feat",
                         disable=not show_progress):
            if np.isnan(x) or np.isnan(y):
                rows.append(np.nan); cols.append(np.nan)
                continue
            c, r = inv * (x, y)
            cols.append(int(floor(c)))
            rows.append(int(floor(r)))

        meta_df = pd.DataFrame({
            "_orig_index": gdf_work.index,
            "_row": rows,
            "_col": cols
        })

        # Group duplicates
        to_remove = set()
        groups = meta_df.dropna().groupby(["_row", "_col"])
        for (r, c), grp in tqdm(groups,
                                desc="Evaluate duplicate pixels",
                                unit="group",
                                disable=not show_progress):
            if len(grp) < 2:
                continue
            idxs = list(grp["_orig_index"])
            if drop == "both":
                to_remove.update(idxs)
            elif drop == "first":
                to_remove.add(idxs[0])
            elif drop == "last":
                to_remove.add(idxs[-1])

        if not to_remove:
            if in_place:
                self.data_layers[layer_name] = gdf
            return gdf, []

        gdf_result = gdf.drop(index=to_remove)
        if in_place:
            self.data_layers[layer_name] = gdf_result

        return gdf_result, sorted(to_remove)
    
    
    
    def add_lon_lat_columns(self, layer_name: str, lon_col: str = "lon", lat_col: str = "lat",
                            show_progress: bool = True) -> gpd.GeoDataFrame:
        """
        Add longitude and latitude columns to a GeoDataFrame layer with a progress bar.
        - Point geometries: direct coordinates.
        - Non-point geometries: centroid coordinates.
        Coordinates are ensured in EPSG:4326.
        """
        if layer_name not in self.data_layers:
            raise ValueError(f"Layer '{layer_name}' not found in data_layers")

        gdf = self.data_layers[layer_name]
        if "geometry" not in gdf.columns:
            raise ValueError(f"Layer '{layer_name}' has no geometry column")

        try:
            gdf_ll = gdf if (gdf.crs and str(gdf.crs).upper() in ("EPSG:4326", "OGC:CRS84")) else gdf.to_crs(4326)
        except Exception:
            gdf_ll = gdf  # assume already lon/lat

        lons = []
        lats = []
        iterator = gdf_ll.geometry
        for geom in tqdm(iterator, desc="Extract lon/lat", unit="feature", disable=not show_progress):
            if geom is None or geom.is_empty:
                lons.append(np.nan); lats.append(np.nan)
            elif geom.geom_type == "Point":
                lons.append(geom.x); lats.append(geom.y)
            else:
                c = geom.centroid
                lons.append(c.x); lats.append(c.y)

        gdf[lon_col] = pd.Series(lons, index=gdf.index, dtype="float64")
        gdf[lat_col] = pd.Series(lats, index=gdf.index, dtype="float64")
        self.data_layers[layer_name] = gdf
        return gdf
    
    
    
    def add_lon_lat_columns_v1(self, layer_name: str, lon_col: str = "lon", lat_col: str = "lat") -> gpd.GeoDataFrame:
        """
        Add longitude and latitude columns to a GeoDataFrame layer.
        - If geometry is a Point: use its coordinates.
        - If geometry is a Polygon/MultiPolygon or any non-point geometry: use centroid.
        Coordinates are computed in EPSG:4326 (lon/lat) and stored in the original layer.

        Parameters:
        - layer_name: name of the layer in data_layers
        - lon_col: name of the longitude column to create
        - lat_col: name of the latitude column to create

        Returns:
        - The modified GeoDataFrame (also updated in self.data_layers)
        """
        if layer_name not in self.data_layers:
            raise ValueError(f"Layer '{layer_name}' not found in data_layers")

        gdf = self.data_layers[layer_name]
        if "geometry" not in gdf.columns:
            raise ValueError(f"Layer '{layer_name}' has no geometry column")

        # Compute lon/lat in EPSG:4326 without changing the stored CRS of the layer
        try:
            gdf_ll = gdf if (gdf.crs and str(gdf.crs).upper() in ("EPSG:4326", "OGC:CRS84")) else gdf.to_crs(4326)
        except Exception:
            # If CRS is missing/invalid, assume it's already lon/lat
            gdf_ll = gdf

        geom = gdf_ll.geometry
        # Masks
        is_valid = geom.notna() & (~geom.is_empty)
        is_point = is_valid & (geom.geom_type == "Point")

        # Prepare output series
        lon = pd.Series(np.nan, index=gdf.index, dtype="float64")
        lat = pd.Series(np.nan, index=gdf.index, dtype="float64")

        # Points: direct coordinates
        if is_point.any():
            lon.loc[is_point] = geom.loc[is_point].x.values
            lat.loc[is_point] = geom.loc[is_point].y.values

        # Non-points: centroid coordinates (covers Polygon, MultiPolygon, LineString, etc.)
        non_point = is_valid & (~is_point)
        if non_point.any():
            cents = geom.loc[non_point].centroid
            lon.loc[non_point] = cents.x.values
            lat.loc[non_point] = cents.y.values

        # Assign to original layer (do not modify original geometry/CRS)
        gdf[lon_col] = lon
        gdf[lat_col] = lat
        self.data_layers[layer_name] = gdf
        return gdf
    
    
    def get_data_layer(self, layer_name):
        return self.data_layers.get(layer_name)
    
    def get_shape(self, layer_name):
        return self.data_layers[layer_name].shape
    
    def set_row(self, layer_name, column_name, row_index, new_value):
        if isinstance(row_index, int):
            self.data_layers[layer_name][column_name].iloc[row_index] = new_value
        self.data_layers[layer_name][column_name].loc[row_index] = new_value
    
    def add_random_series_column(self, layer_name, column_name='random',min=0, max=100, distribution_type='random', mean=0, sd=1):
        if distribution_type == 'random':
            series = pd.Series(np.random.randint(min, max, self.get_shape(layer_name)[0]))
        elif distribution_type == 'standard_normal':
            series = pd.Series(np.random.standard_normal(self.get_shape(layer_name)[0]))
        elif distribution_type == 'normal':
            series = pd.Series(np.random.normal(mean, sd, self.get_shape(layer_name)[0]))
        else:
            series = pd.Series(np.random.randn(self.get_shape(layer_name)[0]))
        self.add_column(layer_name, series, column_name)
    
    def join_layers(self, left_layer, right_layer, on, how='inner'):
        self.data_layers[left_layer] = self.data_layers.get(left_layer).merge(self.data_layers.get(right_layer), on=on, how=how)
    def column_to_list(self, layer_name, column_name, verbose=True):
        column_as_list = self.get_column(layer_name, column_name).tolist()
        if verbose is True:
            print(column_as_list)
        return column_as_list
        
    def add_one_value_column(self, layer_name, column_name, value, length=None):
        """
        Add a column with a single repeated value to the specified GIS layer.
        
        Parameters:
        -----------
        layer_name : str
            Name of the layer to add the column to
        column_name : str
            Name of the new column
        value : any
            Value to fill the column with (can be numeric, string, datetime, etc.)
        length : int, optional
            Length of the column. If None, uses the layer's row count.
            
        Returns:
        --------
        GeoDataFrame
            The modified geodataframe
        """
        if layer_name not in self.data_layers:
            raise ValueError(f"Layer '{layer_name}' not found in data_layers")
        
        # Get the layer's geodataframe
        gdf = self.data_layers[layer_name]
        
        # Determine the length of the column
        if length is None:
            length = len(gdf)
        
        # For non-numeric values like datetime, we can't use np.zeros + fill
        # Instead, create a pandas Series with the repeated value
        gdf[column_name] = pd.Series([value] * length, index=gdf.index)
        
        return gdf
    
    def show(self, layer_name, column4color=None, color=None, alpha=0.5, legend=False, 
             figsize_tuple=(16,9), cmap=None, interactive_mode=False, save_fig=False, savefig_path='out.png', **kwargs):
        """_summary_

        Args:
            layer_name (_type_): _description_
            column4color (_type_, optional): _description_. Defaults to None.
            color (_type_, optional): _description_. Defaults to None.
            alpha (float, optional): _description_. Defaults to 0.5.
            legend (bool, optional): _description_. Defaults to False.
            figsize_tuple (tuple, optional): _description_. Defaults to (15,10).
            cmap (str, optional): example: 'Reds' for heatmaps. Defaults to None.
        """
        
        fig, ax = plt.subplots(figsize=figsize_tuple)
        
        layer = self.data_layers.get(layer_name).to_crs(epsg=3857)
        ax = layer.plot(ax=ax, alpha=alpha, edgecolor='k', color=color, legend=legend, 
                   column=column4color, figsize=figsize_tuple, cmap=cmap, **kwargs)
        cx.add_basemap(ax=ax, source=cx.providers.Esri.WorldImagery, crs=layer.crs.to_string(), attribution=False)
        
        
        if save_fig is True:
            # Adjust the position of the tiles to appear in one line at the bottom
            ax.set_anchor('SW')  # Set the anchor point to the southwest (bottom left) corner
            
            # Adjust the plot limits to cover the entire image
            ax.set_xlim(layer.total_bounds[0], layer.total_bounds[2])
            ax.set_ylim(layer.total_bounds[1], layer.total_bounds[3])
            
            import matplotlib as mpl
            mpl.rcParams['agg.path.chunksize'] = 10000
            fig.savefig(savefig_path, dpi=720, bbox_inches='tight')
                    
        if interactive_mode is True: 
            return self.data_layers.get(layer_name).explore()
        else:
            ax.set_aspect('equal')
            plt.show()
        
    def get_crs(self, layer_name):
        """
        Cordonate Reference System
        EPSG: european petroleum survey group
        """
        return self.get_data_layer(layer_name).crs
    
    def reorder_columns(self, layer_name, new_order_as_list):
        self.data_layers[layer_name].reindex_axis(new_order_as_list, axis=1)
        
    
    def export_dataframe(self, layer_name, destination_path='dataframe.csv', type='csv', index=True):
        """
        Export a GIS layer as a regular DataFrame (without geometry) to a file.
        
        Parameters:
        -----------
        layer_name : str
            Name of the layer to export
        destination_path : str
            Path to save the file
        type : str, default 'csv'
            Format to export ('csv', 'json', or 'pkl')
        index : bool, default True
            Whether to include the index in the output
        """
        if layer_name not in self.data_layers:
            print(f"❌ Error: Layer '{layer_name}' not found")
            return
            
        try:
            # Make a copy so we don't modify the original layer
            df = self.data_layers[layer_name].copy()
            
            # Drop the geometry column to get only the DataFrame
            if 'geometry' in df.columns:
                df = df.drop('geometry', axis=1)
            
            if type == 'json':
                df.to_json(destination_path)
            elif type == 'csv':
                df.to_csv(destination_path, index=index)
            elif type == 'pkl':
                df.to_pickle(destination_path)
            else:
                print(f"❌ Error: Unsupported file format '{type}'")
                return
                
            print(f"✅ Successfully exported layer '{layer_name}' as DataFrame to {destination_path} in {type} format")
            
        except Exception as e:
            print(f"❌ Error exporting DataFrame from layer '{layer_name}': {e}")
    
    def export(self, layer_name, file_name, file_format='geojson'):
        """
        Export a GIS layer to a file in the specified format.
        
        Parameters:
        -----------
        layer_name : str
            Name of the layer to export
        file_name : str
            Path to save the file
        file_format : str, default 'geojson'
            Format to export ('geojson', 'shapefile', or 'parquet')
        """
        if layer_name not in self.data_layers:
            print(f"❌ Error: Layer '{layer_name}' not found")
            return
            
        try:
            if file_format == 'geojson':
                self.data_layers[layer_name].to_file(file_name, driver='GeoJSON')
            elif file_format == 'shapefile':
                self.data_layers[layer_name].to_file(file_name, driver='ESRI Shapefile')
            elif file_format == 'parquet':
                self.data_layers[layer_name].to_parquet(file_name)
            else:
                print(f"❌ Error: Unsupported file format '{file_format}'")
                return
                
            print(f"✅ Successfully exported layer '{layer_name}' to {file_name} in {file_format} format")
            
        except Exception as e:
            print(f"❌ Error exporting layer '{layer_name}': {e}")
            
    def to_crs(self, layer_name, epsg="4326"):
        self.data_layers[layer_name] = self.data_layers[layer_name].to_crs(epsg)
        
    def set_crs(self, layer_name, epsg="4326"):
        self.data_layers[layer_name] = self.data_layers[layer_name].set_crs(epsg)
        
    def show_points(self, x_y_csv_path, crs="4326"):
        pass
    
    def show_point(self, x_y_tuple, crs="4326"):
        pass
    
    def add_point(self, x_y_tuple, layer_name, crs="4326"):
        point = Point(0.0, 0.0)
        #self.__dataframe = self.get_dataframe().append(row_as_dict, ignore_index=True)
        row_as_dict = {'geometry': point}
        self.data_layers[layer_name].append(row_as_dict, ignore_index=True)
    
    def new_data_layer(self, layer_name, crs="EPSG:3857"):
        self.data_layers[layer_name] = gpd.GeoDataFrame(crs=crs)
        self.data_layers[layer_name].crs = crs
        
    
    def merge_geojson_folder(self,
                             folder_path: str,
                             dest_layer_name: str | None = None,
                             pattern: str = "*.geojson",
                             crs: str | None = "auto",
                             add_source: bool = False,
                             show_progress: bool = True):
        """
        Merge all GeoJSON files under a folder into a single GeoDataFrame.

        Parameters
        ----------
        folder_path : str
            Folder containing GeoJSON files.
        dest_layer_name : str | None
            If provided, stores the merged GeoDataFrame at this layer name.
        pattern : str
            Glob pattern to match files (default '*.geojson').
        crs : 'auto' | EPSG string/int | None
            Target CRS for the merged output.
            - 'auto': use the CRS of the first file that has a valid CRS.
            - explicit (e.g., 'EPSG:4326' or 4326): reproject all to this CRS.
            - None: do not reproject (requires all inputs to already match).
        add_source : bool
            If True, adds a 'source_file' column with the filename for each row.
        show_progress : bool
            Show a tqdm progress bar while reading files.

        Returns
        -------
        gpd.GeoDataFrame
            Merged GeoDataFrame.
        """
        import os
        from glob import glob
        import pandas as pd
        import numpy as np
        import geopandas as gpd

        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        files = sorted(glob(os.path.join(folder_path, pattern)))
        if not files:
            raise ValueError(f"No files matching '{pattern}' in {folder_path}")

        target_crs = None
        collected = []

        # Decide target CRS if explicit
        if crs not in (None, "auto"):
            try:
                target_crs = gpd.CRS.from_user_input(crs)
            except Exception as e:
                raise ValueError(f"Invalid CRS '{crs}': {e}")

        for fp in tqdm(files, desc="Read GeoJSONs", unit="file", disable=not show_progress):
            try:
                gdf = gpd.read_file(fp)
            except Exception as e:
                print(f"⚠️ Skip (failed to read): {fp} — {e}")
                continue

            if gdf.empty:
                # Skip empty layers
                continue

            # Determine auto target CRS from first valid file if not set
            if target_crs is None and crs == "auto" and gdf.crs is not None:
                target_crs = gdf.crs

            # Align CRS:
            # - If target_crs is known and gdf has a CRS different → reproject.
            # - If gdf has no CRS and target_crs is known → assume target (set_crs).
            # - If target is None and gdf has CRS → we’ll accept as-is (but may mismatch).
            if target_crs is not None:
                if gdf.crs is None:
                    # Assume it is already in target_crs if missing
                    try:
                        gdf = gdf.set_crs(target_crs, allow_override=True)
                    except Exception:
                        pass
                elif gdf.crs != target_crs:
                    try:
                        gdf = gdf.to_crs(target_crs)
                    except Exception as e:
                        print(f"⚠️ Skip (reproject failed): {fp} — {e}")
                        continue

            if add_source:
                gdf["source_file"] = os.path.basename(fp)

            collected.append(gdf)

        if not collected:
            raise ValueError("No valid GeoDataFrames collected.")

        # Unify columns (preserve 'geometry')
        unified_cols = set()
        for gdf in collected:
            unified_cols.update(gdf.columns)

        unified_cols = list(unified_cols)
        # Ensure geometry column appears at the end
        if "geometry" in unified_cols:
            unified_cols = [c for c in unified_cols if c != "geometry"] + ["geometry"]

        prepared = []
        for gdf in collected:
            # Add missing columns as NaN (except geometry)
            missing = [c for c in unified_cols if c not in gdf.columns and c != "geometry"]
            for m in missing:
                gdf[m] = pd.Series([np.nan] * len(gdf), index=gdf.index)
            # Reorder to unified order (keep geometry if present)
            if "geometry" in gdf.columns:
                gdf = gdf[[c for c in unified_cols if c in gdf.columns]]
            else:
                gdf = gdf[[c for c in unified_cols if c != "geometry"]]
            prepared.append(gdf)

        merged_pd = pd.concat(prepared, ignore_index=True)
        merged = gpd.GeoDataFrame(merged_pd, geometry="geometry" if "geometry" in merged_pd.columns else None, crs=target_crs)

        if dest_layer_name:
            self.data_layers[dest_layer_name] = merged
            print(f"✅ Merged {len(collected)} file(s) → layer '{dest_layer_name}' with {len(merged)} rows.")

        return merged
    
    
    def add_column(self, layer_name, column, column_name):
        y = column
        if (not isinstance(column, pd.core.series.Series or not isinstance(column, pd.core.frame.DataFrame))):
            y = np.array(column)
            y = np.reshape(y, (y.shape[0],))
            y = pd.Series(y)
        self.data_layers[layer_name][column_name] = y
        
    def show_dataframe(self, layer_name, number_of_row=None):
        if number_of_row is None:
            print(self.get_data_layer(layer_name))
        elif number_of_row < 0:
            return self.get_data_layer(layer_name).tail(abs(number_of_row)) 
        else:
            return self.get_data_layer(layer_name).head(number_of_row) 
        
    def add_row(self, layer_name, row_as_dict):
        self.data_layers[layer_name] = self.get_data_layer(layer_name).append(row_as_dict, ignore_index=True)
    
    def get_row(self, layer_name, row_index, column=None):
        if column is not None:
            return self.data_layers[layer_name].loc[self.data_layers[layer_name][column] == row_index].reset_index(drop=True)
        return self.data_layers[layer_name].iloc[row_index]
    
    def get_layer_shape(self, layer_name):
        """
        return (Number of lines, number of columns)
        """
        return self.data_layers[layer_name].shape
    
    def get_columns_names(self, layer_name):
        header = list(self.data_layers[layer_name].columns)
        return header 
    
    def drop_column(self, layer_name, column_name):
        """Drop a given column from the dataframe given its name

        Args:
            column (str): name of the column to drop

        Returns:
            [dataframe]: the dataframe with the column dropped
        """
        self.data_layers[layer_name] = self.data_layers[layer_name].drop(column_name, axis=1)
        return self.data_layers[layer_name]
    
    def keep_columns(self, layer_name, columns_names_as_list):
        for p in self.get_columns_names(layer_name):
            if p not in columns_names_as_list:
                self.data_layers[layer_name] = self.data_layers[layer_name].drop(p, axis=1)
    def group_by(self, layer_name, group_by_column_name, agg_func='sum'):
        # Dissolve the fields by the 'canal' column
        self.data_layers[layer_name] = self.data_layers[layer_name].dissolve(by=group_by_column_name, aggfunc='sum')
                
    def get_area_column(self, layer_name):
        return self.get_data_layer(layer_name).area
    
    def get_perimeter_column(self, layer_name):
        return self.get_data_layer(layer_name).length
    
    def get_row_area(self, layer_name, row_index):
        return self.data_layers[layer_name].area.iloc[row_index]
    
    def get_distance(self, layer_name, index_column, row_index_a, row_index_b):
        if 1 == 1:
            other = self.get_row(layer_name, row_index_b, index_column)
            return self.get_row(layer_name, row_index_a, index_column).distance(other)
    
    def filter_dataframe(self, layer_name, column, func_de_decision, in_place=True, *args):
        if in_place is True:
            if len(args) == 2:
                self.set_dataframe(
                    self.data_layers[layer_name].loc[self.get_column(column).apply(func_de_decision, args=(args[0], args[1]))])
            else:
                self.set_dataframe(
                    self.data_layers[layer_name].loc[self.get_column(column).apply(func_de_decision)])
        else:
            if len(args) == 2:
                return self.data_layers[layer_name].loc[self.get_column(column).apply(func_de_decision, args=(args[0], args[1]))]
            else:
                return self.data_layers[layer_name].loc[self.get_column(column).apply(func_de_decision)]

    def transform_column(self, layer_name, column_to_trsform, column_src, fun_de_trasformation, in_place= True,*args):
        if in_place is True:
            if (len(args) != 0):
                self.set_column(layer_name, column_to_trsform, self.get_column(layer_name, column_src).apply(fun_de_trasformation, args=(args[0],)))
            else:
                self.set_column(layer_name, column_to_trsform, self.get_column(layer_name, column_src).apply(fun_de_trasformation)) 
        else:
            if (len(args) != 0):
                return self.get_column(layer_name, column_src).apply(fun_de_trasformation, args=(args[0],))
            else:
                return self.get_column(layer_name, column_src).apply(fun_de_trasformation)
            
    def set_column(self, layer_name, column_name, new_column):
        self.data_layers[layer_name][column_name] = new_column
    
    def get_column(self, layer_name, column_name):
        return self.data_layers[layer_name][column_name]
    
    def reindex_dataframe(self, layer_name, index_as_liste=None, index_as_column_name=None):
        if index_as_liste is not None:
            new_index = new_index = index_as_liste
            self.data_layers[layer_name].index = new_index
        if index_as_column_name is not None:
            self.data_layers[layer_name].set_index(index_as_column_name, inplace=True)
        if index_as_column_name is None and index_as_liste is None:
            new_index = pd.Series(np.arange(self.get_shape()[0]))
            self.data_layers[layer_name].index = new_index
            
    def get_era5_land_grib_as_dataframe(self, file_path, layer_name):
        grip_path = file_path
        ds = xr.load_dataset(grip_path, engine="cfgrib")
        self.data_layers[layer_name] = DataFrame()
        self.data_layers[layer_name].set_dataframe(ds.to_dataframe())
        return ds.to_dataframe()
    
    def rename_columns(self, layer_name, column_dict_or_all_list, all_columns=False):
        if all_columns is True:
            types = {}
            self.data_layers[layer_name].columns = column_dict_or_all_list
            for p in column_dict_or_all_list:
                types[p] = str
            self.data_layers[layer_name] = self.get_dataframe().astype(types)
        else:
            self.data_layers[layer_name].rename(columns=column_dict_or_all_list, inplace=True)
            
    def add_area_column(self, layer_name, unit='ha'):
        self.add_column(layer_name, self.get_area_column(layer_name), 'area')
        if unit == 'ha':
            self.add_column(layer_name, self.get_area_column(layer_name) / 10000, 'area')
            
    def index_to_column(self, layer_name, column_name=None, drop_actual_index=False, **kwargs):
        self.data_layers[layer_name].reset_index(drop=drop_actual_index, inplace=True, **kwargs) 
        if column_name is not None:
            self.rename_columns({'index': column_name})
            
        
    def calculate_perimeter_as_column(self, layer_name):
        self.add_column(layer_name, self.get_perimeter_column(layer_name), 'perimeter')
        
    def count_occurrence_of_each_row(self, layer_name, column_name):
        return self.data_layers[layer_name].pivot_table(index=[column_name], aggfunc='size')
    
    def count_occurrence_of_each_row(self, layer_name, column_name):
        return self.data_layers[layer_name].pivot_table(index=[column_name], aggfunc='size')
    
    def add_transformed_columns(self, layer_name, dest_column_name="new_column", transformation_rule="okk*2"):
        columns_names = self.get_columns_names(layer_name)
        columns_dict = {}
        for column_name in columns_names:
            if column_name in transformation_rule:
                columns_dict.update({column_name: self.get_column(layer_name, column_name)})
        y_transformed = eval(transformation_rule, columns_dict)
        self.data_layers[layer_name][dest_column_name] = y_transformed
    
    @staticmethod
    def lon_lat_dataframe_to_geopandas(src_dataframe, lon_column_name, lat_column_name, crs=4326):
        from pyproj import CRS
        # Assuming your GeoDataFrame is named 'gdf' and you want to reproject to EPSG:4326
        target_crs = CRS.from_epsg(crs)

        geometry = [Point(lon, lat) for lon, lat in zip(src_dataframe[lon_column_name], src_dataframe[lat_column_name])]
        geo_data = gpd.GeoDataFrame(src_dataframe, geometry=geometry, crs=target_crs)
        
        # Reproject the GeoDataFrame
        gdf_reprojected = geo_data.to_crs(target_crs)
        
        return gdf_reprojected

    @staticmethod
    def new_geodaraframe_from_points():
        map.new_data_layer('valves', crs="ESRI:102191")
        for p in range(map.get_layer_shape('pipelines')[0]):
            #print(map.get_row('pipelines', p))
            vi = Point(map.get_row('pipelines', p)['X_Start'], map.get_row('pipelines', p)['Y_Start'])
            vf = Point(map.get_row('pipelines', p)['X_End'], map.get_row('pipelines', p)['Y_End'])
            id_pipeline = map.get_row('pipelines', p)['Nom_CANAL']
            row_as_dict = {'id_pipeline': id_pipeline,
                        'geometry': vi}
            map.add_row('valves', row_as_dict)
            row_as_dict = {'id_pipeline': id_pipeline,
                        'geometry': vf}
            map.add_row('valves', row_as_dict)
        
    
        
    