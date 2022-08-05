import copy
import xarray as xr

import numpy as np
from numba import jit, float32, int32
from math import atan2, ceil, pi, log, sqrt, pow, fabs, cos, sin
from skimage.measure import regionprops, label
from .object_identification import label
from .object_quality_control import QualityControler, whereeq
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
from time import time

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

#ANALYSIS_DX = 3000

def get_constituent_storms(
	model,
	min_thresh,
	dbzcomp_qc_params, 
	storm_types,
	dbzcomp_labels,
	dbzcomp_props,
	dbzcomp_values,
	uh_labels,
	uh_props,
	uh_values,
	itr,
	max_itr,
	ANALYSIS_DX, 
	debug=False,
):
	new_storm_types = storm_types.copy()
	new_dbzcomp_labels = dbzcomp_labels.copy()
	new_dbzcomp_props = dbzcomp_props.copy()

	label_inc = 100 * (itr - 1)
	
	t1 = time()
	temp_dbzcomp_labels, temp_dbzcomp_props = label(
		input_data=dbzcomp_values,
		method="single_threshold",
		params={"bdry_thresh": min_thresh},
	)
	
	if len(temp_dbzcomp_props) > 0:
		# Apply quality control
		t1 = time()
		temp_dbzcomp_labels, temp_dbzcomp_props = QualityControler().quality_control(
			object_labels=temp_dbzcomp_labels,
			object_properties=temp_dbzcomp_props,
			input_data=dbzcomp_values,
			qc_params=dbzcomp_qc_params,
		)

	# Identify candidate embedded storm objects
	t1 = time()
	temp_storm_types, labels_with_matched_rotation = get_storm_types(
		model,
		temp_dbzcomp_labels,
		temp_dbzcomp_props,
		dbzcomp_values,
		uh_labels,
		uh_props,
		uh_values,
		ANALYSIS_DX, 
		itr == max_itr,
	)
	
	dbz_coords = [prop.coords for prop in dbzcomp_props]
	if (len(temp_storm_types)):
		new_storm_types,new_dbzcomp_labels,n_append = iterate_storm_types(
						storm_types, new_storm_types,new_dbzcomp_labels,
						dbz_coords, 
						[prop.centroid for prop in dbzcomp_props],
						[prop.equivalent_diameter for prop in dbzcomp_props], 
						[prop.area for prop in dbzcomp_props],
						temp_storm_types, 
						[prop.coords for prop in temp_dbzcomp_props], 
						[prop.label for prop in temp_dbzcomp_props],
						temp_dbzcomp_labels,
						[prop.centroid for prop in temp_dbzcomp_props], 
						[prop.equivalent_diameter for prop in temp_dbzcomp_props],
						[prop.area for prop in temp_dbzcomp_props],
						label_inc
						)
						
		for n in n_append:
			new_dbzcomp_props.append(temp_dbzcomp_props[n])

	# After last iteration, do final QC

	if itr == max_itr:

		# Create "family tree" of objects; new_storm_depths contains the "generation" of each object; 
		# new_storm_parent_labels contains the parent label for each child object; new_storm_embs 
		# contains the parent type of each child object
		# new_storm_depths is later used to process objects in order of their generation

		new_storm_parent_labels, new_storm_embs, new_storm_depths = object_hierarchy(
			new_storm_types, new_dbzcomp_props
		)

		new_storm_types2 = new_storm_types.copy()
		new_storm_embs2 = new_storm_embs.copy()
		new_dbzcomp_props2 = new_dbzcomp_props.copy()
		new_dbzcomp_labels2 = new_dbzcomp_labels.copy()
		new_storm_parent_labels2 = new_storm_parent_labels.copy()
		new_storm_depths2 = new_storm_depths.copy()

	else:

		new_storm_embs = ["NONEMB" for x in new_storm_types]

	if itr == max_itr:

		remove_indices = []
		parent_labels = []

		for n, storm_type in enumerate(new_storm_types):

			if new_storm_embs[n] in ["EMB-QLCS"]:

				"""parent_label = new_dbzcomp_props[n].label

				for nn in range(len(new_storm_types)):
				  if nn != n and new_storm_parent_labels[nn] == parent_label:
					if nn not in remove_indices:
					  remove_indices.append(nn)
					  print ("Removing %s within %s within QLCS" % (new_storm_types[nn], storm_type))"""

				parent_coords = np.asarray(new_dbzcomp_props[n].coords)

				for nn, storm2_region in enumerate(new_dbzcomp_props):

					if new_dbzcomp_props[nn].area < new_dbzcomp_props[n].area:

						child_coords = np.asarray(storm2_region.coords)

						if check_overlap(child_coords, parent_coords):

							if nn not in remove_indices:
								remove_indices.append(nn)
								parent_labels.append(new_storm_parent_labels[nn])
								if debug:
									print(
										"Removing %s within %s within %s"
										% (
											new_storm_types[nn],
											storm_type,
											new_storm_embs[n][4:],
										)
									)

		for ind in sorted(remove_indices, reverse=True):
			new_storm_types2.pop(ind)
			new_storm_embs2.pop(ind)
			new_dbzcomp_props2.pop(ind)
			new_storm_parent_labels2.pop(ind)

		new_storm_parent_labels2, new_storm_embs2, new_storm_depths2 = object_hierarchy(
			new_storm_types2, new_dbzcomp_props2
		)

		new_storm_types = new_storm_types2.copy()
		new_storm_embs = new_storm_embs2.copy()
		new_dbzcomp_props = new_dbzcomp_props2.copy()
		new_dbzcomp_labels = new_dbzcomp_labels2.copy()
		new_storm_parent_labels = new_storm_parent_labels2.copy()
		new_storm_depths = new_storm_depths2.copy()

	if itr == max_itr:

		# Reclassify QLCS as cluster if object area dominated by CELLs

		for n, storm_type in enumerate(new_storm_types):

			if (
				new_storm_types[n] == "QLCS"
				and new_dbzcomp_props[n].major_axis_length < 150 / 3.0
			):

				num_const = 0
				num_rot = 0
				const_area = 0

				for nn in range(len(new_storm_types)):

					if (
						nn != n
						and new_storm_parent_labels[nn] == new_dbzcomp_props[n].label
					):

						num_const += 1
						if new_storm_types[nn] == "ROTATE":
							num_rot += 1
						const_area += new_dbzcomp_props[nn].area

				const_area_frac = const_area / float(new_dbzcomp_props[n].area)

				if const_area_frac > 0.75:

					if num_rot > 1:
						new_storm_types2[n] = "SUP_CLUST"
					elif num_const > 1:
						new_storm_types2[n] = "CLUSTER"
					else:
						new_storm_types2[n] = "AMORPHOUS"
					if debug:
						print(
							"CONVERTING QLCS to %s since const_area_frac = %.2f"
							% (new_storm_types2[n], const_area_frac)
						)

				else:

					if debug:
						print(
							"RETAINING QLCS since const_area_frac = %.2f"
							% const_area_frac
						)

		new_storm_parent_labels2, new_storm_embs2, new_storm_depths2 = object_hierarchy(
			new_storm_types2, new_dbzcomp_props2
		)

		new_storm_types = new_storm_types2.copy()
		new_storm_embs = new_storm_embs2.copy()
		new_dbzcomp_props = new_dbzcomp_props2.copy()
		new_dbzcomp_labels = new_dbzcomp_labels2.copy()
		new_storm_parent_labels = new_storm_parent_labels2.copy()
		new_storm_depths = new_storm_depths2.copy()

	if itr == max_itr + 999:

		remove_indices = []

		for n, storm_type in enumerate(new_storm_types):

			if new_storm_embs[n] in ["EMB-QLCS"]:

				"""parent_label = new_dbzcomp_props[n].label

				for nn in range(len(new_storm_types)):
				  if nn != n and new_storm_parent_labels[nn] == parent_label:
					if nn not in remove_indices:
					  remove_indices.append(nn)
					  print ("Removing %s within %s within QLCS" % (new_storm_types[nn], storm_type))"""

				parent_coords = np.asarray(new_dbzcomp_props[n].coords)

				for nn, storm2_region in enumerate(new_dbzcomp_props):

					if new_dbzcomp_props[nn].area < new_dbzcomp_props[n].area:

						child_coords = np.asarray(storm2_region.coords)

						if check_overlap(child_coords, parent_coords):

							if nn not in remove_indices:
								remove_indices.append(nn)
								if debug:
									print(
										"Removing %s within %s within %s"
										% (
											new_storm_types[nn],
											storm_type,
											new_storm_embs[n][4:],
										)
									)

		for ind in sorted(remove_indices, reverse=True):
			new_storm_types2.pop(ind)
			new_storm_embs2.pop(ind)
			new_dbzcomp_props2.pop(ind)
			new_storm_parent_labels2.pop(ind)

		new_storm_parent_labels2, new_storm_embs2, new_storm_depths2 = object_hierarchy(
			new_storm_types2, new_dbzcomp_props2
		)

		new_storm_types = new_storm_types2.copy()
		new_storm_embs = new_storm_embs2.copy()
		new_dbzcomp_props = new_dbzcomp_props2.copy()
		new_dbzcomp_labels = new_dbzcomp_labels2.copy()
		new_storm_parent_labels = new_storm_parent_labels2.copy()
		new_storm_depths = new_storm_depths2.copy()

	if itr == max_itr and len(new_storm_depths)>0:

		# If object embedded within QLCS_embedded object, discard the former
		# If object embedded within AMORPHOUS or CELL is the only object embedded therein, discard it. If parent object is AMORPHOUS, reclassify it as discarded child object type.
		# If object embedded within AMORPHOUS or CELL is one of multiple such objects, and parent object is unembedded, reclassify parent as CLUSTER or SUP_CLUSTER.
		# If object embedded within AMORPHOUS or CELL is one of multiple such objects, and parent object is embedded in another object, remove parent since it we don't want embedded clusters

		for depth in range(max(new_storm_depths), -1, -1):

			if debug:
				print("depth = %d" % depth)
			remove_indices = []
			parent_labels = []

			for n, storm_type in enumerate(new_storm_types):

				if new_storm_depths[n] == depth:  # and depth==2:

					if new_storm_embs[n] in ["EMB-QLCS999"]:

						parent_coords = np.asarray(new_dbzcomp_props[n].coords)

						for nn, storm2_region in enumerate(new_dbzcomp_props):

							if new_dbzcomp_props[nn].area < new_dbzcomp_props[n].area:

								child_coords = np.asarray(storm2_region.coords)

								if check_overlap(child_coords, parent_coords):

									if nn not in remove_indices:
										remove_indices.append(nn)
										parent_labels.append(new_dbzcomp_props[n].label)
										if debug:
											print(
												"Removing %s within %s within %s"
												% (
													new_storm_types[nn],
													storm_type,
													new_storm_embs[n][4:],
												)
											)

					elif new_storm_embs[n] in ["EMB-CELL", "EMB-CLUS"]:

						parent_label = new_storm_parent_labels[n]
						const_indices = [n]

						for nn in range(len(new_storm_types)):
							if new_dbzcomp_props[nn].label == parent_label:
								parent_index = nn
								break

						if new_storm_types[n] == "ROTATE":
							num_sups = 1
						else:
							num_sups = 0

						for nn in range(len(new_storm_types)):

							if nn != n and new_storm_parent_labels[nn] == parent_label:

								const_indices.append(nn)
								if new_storm_types[nn] == "ROTATE":
									num_sups += 1

						if len(const_indices) == 1:
							if n not in remove_indices:
								remove_indices.append(n)
								parent_labels.append(parent_label)
								if new_storm_embs[n] == "EMB-CLUS":
									new_storm_types2[parent_index] = storm_type
								if debug:
									print(
										"Removing singleton %s from %s (which is now a %s)"
										% (
											storm_type,
											new_storm_types[parent_index],
											storm_type,
										)
									)

						elif new_storm_embs[
							parent_index
						] == "NONEMB" and new_storm_types[parent_index] not in [
							"CLUSTER",
							"SUP_CLUST",
						]:

							if num_sups > 1:
								if debug:
									print(
										"Converting parent %s to SUP_CLUST"
										% new_storm_types2[parent_index]
									)
								new_storm_types2[parent_index] = "SUP_CLUST"
							else:
								if debug:
									print(
										"Converting parent %s to CLUSTER"
										% new_storm_types2[parent_index]
									)
								new_storm_types2[parent_index] = "CLUSTER"
							for ind in const_indices:
								new_storm_embs2[ind] = "EMB-CLUS"

						elif new_storm_embs[parent_index] != "NONEMB":

							if parent_index not in remove_indices:
								if debug:
									print(
										"Removing parent %s since it is actually an embedded cluster"
										% new_storm_types[parent_index]
									)
								remove_indices.append(parent_index)
								parent_labels.append(
									new_storm_parent_labels[parent_index]
								)
								for ind in const_indices:
									new_storm_embs2[ind] = "EMB-CLUS"

			if len(remove_indices) != len(set(remove_indices)):
				if debug:
					print("DUPLICATES in remove_indices!!! Exiting.")
					sys.exit(1)

			for ind in sorted(remove_indices, reverse=True):
				new_storm_types2.pop(ind)
				new_storm_embs2.pop(ind)
				new_dbzcomp_props2.pop(ind)
				new_storm_parent_labels2.pop(ind)

			(
				new_storm_parent_labels2,
				new_storm_embs2,
				new_storm_depths2,
			) = object_hierarchy(new_storm_types2, new_dbzcomp_props2)

			new_storm_types = new_storm_types2.copy()
			new_storm_embs = new_storm_embs2.copy()
			new_dbzcomp_props = new_dbzcomp_props2.copy()
			new_dbzcomp_labels = new_dbzcomp_labels2.copy()
			new_storm_parent_labels = new_storm_parent_labels2.copy()
			new_storm_depths = new_storm_depths2.copy()

	if itr == max_itr:

		for n, storm_type in enumerate(new_storm_types):

			if new_storm_types[n] == "CLUSTER":

				num_sup = 0
				parent_coords = np.asarray(new_dbzcomp_props[n].coords)

				for nn, storm2_region in enumerate(new_dbzcomp_props):

					if (
						new_storm_types[nn] == "ROTATE"
						and new_dbzcomp_props[nn].area < new_dbzcomp_props[n].area
					):

						child_coords = np.asarray(storm2_region.coords)

						if check_overlap(child_coords, parent_coords):

							num_sup += 1

							if num_sup == 2:
								new_storm_types[n] = "SUP_CLUST"
								if debug:
									print("CONVERTING CLUSTER TO SUP_CLUST!!! Exiting.")
									sys.exit(1)
								break

		new_storm_parent_labels2, new_storm_embs2, new_storm_depths2 = object_hierarchy(
			new_storm_types2, new_dbzcomp_props2
		)

		new_storm_types = new_storm_types2.copy()
		new_storm_embs = new_storm_embs2.copy()
		new_dbzcomp_props = new_dbzcomp_props2.copy()
		new_dbzcomp_labels = new_dbzcomp_labels2.copy()
		new_storm_parent_labels = new_storm_parent_labels2.copy()
		new_storm_depths = new_storm_depths2.copy()

	if itr == max_itr:

		# Recast irregular discrete cell objects as AMORPHOUS

		new_storm_types2 = list(new_storm_types)
		new_storm_embs2 = list(new_storm_embs)
		new_dbzcomp_props2 = list(new_dbzcomp_props)
		new_dbzcomp_labels2 = new_dbzcomp_labels.copy()

		for n, storm_type in enumerate(new_storm_types):

			if (
				storm_type in ["ROTATE", "NONROT"]
				and new_storm_embs[n] == "NONEMB"
				and (
					new_dbzcomp_props[n].major_axis_length > 75 / 3.0
					or new_dbzcomp_props[n].solidity < 0.7
					or new_dbzcomp_props[n].eccentricity > 0.97
				)
			):

				new_storm_types2[n] = "AMORPHOUS"
				if debug:
					print(
						"Converting DISCRETE CELL to AMORPHOUS",
						new_dbzcomp_props[n].area,
						new_dbzcomp_props[n].major_axis_length,
						new_dbzcomp_props[n].eccentricity,
						new_dbzcomp_props[n].solidity,
					)

		new_storm_types = new_storm_types2
		new_storm_embs = new_storm_embs2
		new_dbzcomp_props = new_dbzcomp_props2
		new_dbzcomp_labels = new_dbzcomp_labels2

	if itr == max_itr:
		return new_storm_types, new_storm_embs, new_dbzcomp_props, new_storm_depths2
	else:
		return new_storm_types, new_storm_embs, new_dbzcomp_props, None

	return new_storm_types, new_storm_embs, new_dbzcomp_labels, new_dbzcomp_props
  
  
@jit(fastmath=True) 
def iterate_storm_types(storm_types, new_storm_types, new_dbzcomp_labels, 
						dbz_coords,dbz_centroid,dbz_equiv_diam,dbz_area,
						temp_storm_types, temp_coords,temp_prop_label,temp_dbzcomp_labels,
						temp_centroid,temp_equiv_diam, temp_area, label_inc): 
	
	new_inds = []	
	for n, storm_type in enumerate(storm_types):
		
		# Cycle through candidate storm objects

		prelim_new_storm_types = []
		prelim_new_dbzcomp_centroid = []
		prelim_new_dbzcomp_equiv_diam= []
		prelim_new_dbzcomp_area = []
		prelim_temp_labels = []
		temp_inds = []

		parent_coords = dbz_coords[n]
		
		for nn, prop in enumerate(temp_prop_label):
			child_coords = temp_coords[nn]
			if temp_storm_types[nn] in ["ROTATE", "NONROT"]:
				
				if check_overlap(child_coords, parent_coords):
					# If candidate storm object overlaps an existing storm object,
					# retain and characterize it for further testing
					prelim_new_storm_types.append(temp_storm_types[nn])
					temp_prop_label[
						nn
					] += label_inc	# ensure new object has unique label so as not to combine with preexisting object
					prelim_new_dbzcomp_centroid.append(temp_centroid[nn])
					prelim_new_dbzcomp_area.append(temp_area[nn])
					prelim_new_dbzcomp_equiv_diam.append(temp_equiv_diam[nn])
					prelim_temp_labels.append(temp_prop_label[nn])
					temp_inds.append(nn)

		# cycle through each candidate new storm object
		for nnn, new_storm_type in enumerate(prelim_new_storm_types):

			# if storm object is a CELL, determine if this object is likely just a smaller 
			# version of an existing object (versus a constituent of a larger storm)

			if new_storm_type in ["ROTATE", "NONROT"]:

				similar_storm_found = False

				prelim_i = prelim_new_dbzcomp_centroid[nnn][0]
				prelim_j = prelim_new_dbzcomp_centroid[nnn][1]

				for nnnn, exist_props in enumerate(dbz_centroid):

					if new_storm_types[nnnn] in ["ROTATE", "NONROT"]:  ### NEW

						exist_i = exist_props[0]
						exist_j = exist_props[1]
						dist = sqrt(
							pow(prelim_i - exist_i, 2) + pow(prelim_j - exist_j, 2)
						)

						if (
							dist < max(2, 0.15 * dbz_equiv_diam[nnn])
							and prelim_new_dbzcomp_area[nnn] <= 1.0 * dbz_area[nnnn]
						):

							similar_storm_found = True
							break

				# if candidate object is not merely a smaller version of an existing object, retain it

				if not similar_storm_found:
					new_storm_types.append(prelim_new_storm_types[nnn])
					new_inds.append(temp_inds[nnn])
					dbz_centroid.append(prelim_new_dbzcomp_centroid[nnn])
					dbz_area.append(prelim_new_dbzcomp_area[nnn])
					dbz_equiv_diam.append(prelim_new_dbzcomp_equiv_diam[nnn])
					new_dbzcomp_labels = whereeq(new_dbzcomp_labels.astype(np.int64),temp_dbzcomp_labels.astype(np.int64),
					np.int64(prelim_temp_labels[nnn] - label_inc),np.int64(prelim_temp_labels[nnn])).astype(np.int8)
					
	return new_storm_types,new_dbzcomp_labels,new_inds

 
def get_storm_labels(storm_emb, storm_type):

	convert_labels = {
		"ROTATE": "SUPERCELL",
		"NONROT": "ORDINARY",
		"SEGMENT": "OTHER",
		"QLCS": "QLCS",
		"CLUSTER": "OTHER",
		"AMORPHOUS": "OTHER",
		"QLCS_ROT": "QLCS_MESO",
		"QLCS_NON": "QLCS_ORD",
		"CLUS_ROT": "SUPERCELL",
		"CLUS_NON": "ORDINARY",
		"SUP_CLUST": "SUP_CLUST",
		"CELL_NON": "ORDINARY",
		"CELL_ROT": "SUPERCELL",
	}
	digitize_types = {
		key: n
		for n, key in enumerate(
			[
				"ORDINARY",
				"SUPERCELL",
				"QLCS",
				"SUP_CLUST",
				"QLCS_ORD",
				"QLCS_MESO",
				"OTHER",
			]
		)
	}

	if storm_emb == "NONEMB":
		label = storm_type
	else:
		label = "%s_%s" % (storm_emb[4:], storm_type[:3])

	type_str = convert_labels[label]
	type_int = int(digitize_types[type_str])

	return type_int, type_str


def get_storm_types(
	model,
	qc_dbzcomp_labels,
	qc_dbzcomp_props,
	dbzcomp,
	qc_uh_labels,
	qc_uh_props,
	uh,
	ANALYSIS_DX, 
	last_iter=False,
):
	"""
	# PURPOSE:

	# Classify storm objects based on REFLCOMP and UH5000/AZSHR fields

	# INPUTS:

	# model (str), date (str), lead_time (int; mins) for which qc_dbzcomp_labels, etc. are valid

	# qc_dbzcomp_labels: single QC'd REFLCOMP 2-D label numpy array
	# qc_dbzcomp_props: regionprops for each label in qc_dbzcomp_labels
	# dbzcomp: 2-D REFLCOMP field from which qc_dbzcomp_labels, qc_dbzcomp_props generated

	# uh_labels/qc_uh_props/uh: as with qc_dbzcomp_labels/qc_dbzcomp_props/dbzcomp

	# RETURNS:

	# List of storm types (one for each regionprops in qc_dbzcomp_props, i.e., one for each REFLCOMP object)

	# Author: Corey Potvin
	"""
	
	storm_types = []
	labels_with_matched_rotation = []
	all_uh_coords = []

	# Concatenate all coordinates of strong rotation, then identify regions with proximate rotation objects

	for n, qc_uh_prop in enumerate(qc_uh_props):
		for coord in qc_uh_prop.coords:
			all_uh_coords.append(coord)
	all_uh_coords = np.asarray(all_uh_coords)

	if len(all_uh_coords) > 0:
		for qc_dbzcomp_prop in qc_dbzcomp_props:
			ic, jc = int(qc_dbzcomp_prop.centroid[0]), int(qc_dbzcomp_prop.centroid[1])
			rad = max(2, ceil(qc_dbzcomp_prop.equivalent_diameter / 3.0))
			imin, imax = ic - rad, ic + rad
			jmin, jmax = jc - rad, jc + rad
			dbz_coords = []
			for i in range(imin, imax + 1):
				for j in range(jmin, jmax + 1):
					dbz_coords.append([i, j])
			dbz_coords = np.asarray(dbz_coords)
			if check_overlap(dbz_coords, all_uh_coords):
				labels_with_matched_rotation.append(qc_dbzcomp_prop.label)

	for n, region in enumerate(qc_dbzcomp_props):

		# see scikit-image regionprops documentation for definitions of object attributes

		DBZ_length = region.major_axis_length * ANALYSIS_DX
		DBZ_area = region.area * pow(ANALYSIS_DX, 2)

		if DBZ_area < 100e6 or (
			DBZ_area < 200e6 and region.eccentricity < 0.7 and region.solidity > 0.7
		):

			if region.label in labels_with_matched_rotation:
				storm_type = "ROTATE"
			else:
				storm_type = "NONROT"

		elif region.eccentricity > 0.97:

			if DBZ_length > 75e3:
				storm_type = "QLCS"
			else:
				storm_type = "AMORPHOUS"

		elif region.eccentricity > 0.9 and DBZ_length > 150e3:
			storm_type = "QLCS"

		elif (
			region.eccentricity > 0.93
			or DBZ_area > 500e6
			or DBZ_length > 75e3
			or region.solidity < 0.7
		):
			storm_type = "AMORPHOUS"

		elif region.label in labels_with_matched_rotation:
			storm_type = "ROTATE"

		else:
			storm_type = "NONROT"

		# if storm_type in ['ROTATE', 'NONROT'] and (DBZ_length > 75e3 or region.solidity < 0.7 or region.eccentricity > 0.97):
		#  storm_type = 'AMORPHOUS'

		storm_types.append(storm_type)

	return storm_types, labels_with_matched_rotation

@jit
def check_overlap(coords1, coords2):

	for item1 in coords1:
		for item2 in coords2:
			if (item1 == item2).all():
				return True

	return False


def object_hierarchy(storm_types, dbzcomp_props):

	storm_parent_labels = list(np.zeros(len(storm_types)))
	storm_embs = list(np.zeros(len(storm_types)))

	for n, storm_type in enumerate(storm_types):

		child_coords = np.asarray(dbzcomp_props[n].coords)
		min_potential_parent_area = 9999999999
		parent_found = False

		for nn in range(len(dbzcomp_props)):
			potential_parent_area = dbzcomp_props[nn].area
			if potential_parent_area > dbzcomp_props[n].area:
				potential_parent_coords = np.asarray(dbzcomp_props[nn].coords)
				if check_overlap(child_coords, potential_parent_coords):
					parent_found = True
					if potential_parent_area < min_potential_parent_area:
						min_potential_parent_area = potential_parent_area
						potential_parent_index = nn

		if parent_found:
			storm_parent_labels[n] = dbzcomp_props[potential_parent_index].label
			if storm_types[potential_parent_index] == "QLCS":
				storm_embs[n] = "EMB-QLCS"
			elif storm_types[potential_parent_index] in [
				"CLUSTER",
				"AMORPHOUS",
				"SUP_CLUST",
			]:
				storm_embs[n] = "EMB-CLUS"
			else:
				storm_embs[n] = "EMB-CELL"
		else:
			storm_embs[n] = "NONEMB"

		storm_labels = [props.label for props in dbzcomp_props]

	storm_depths = []

	for n in range(len(storm_parent_labels)):

		depth = 0
		nn = n

		while storm_parent_labels[nn] > 0:

			depth += 1
			parent_label = storm_parent_labels[nn]
			nn = storm_labels.index(parent_label)

		storm_depths.append(depth)
		# print (n, storm_types[n], depth)

	return storm_parent_labels, storm_embs, storm_depths
