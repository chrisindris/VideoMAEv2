while read p; do
	left=$(find /data/i5O/UCF101-THUMOS/THUMOS14/ -name "*${p}.mp4")
	echo $left
done <actionformer_subset.txt
