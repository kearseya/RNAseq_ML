# Log files

Genereated with script (graphing commands were excluded to allow for continuous running of the script):

```bash
for i in {1..30};
	time python3 -u test_script.py > logs/test_$i.log;
done
```

The STDERR containing the time data was manually copied and pasted into the time1_30 file, the real times were then extracted using:

```bash
cat time1_30 | grep "real" | sed -e 's/real\t//g' | awk -F 'm' '{sum += $1*60 + $2} END { print sum/60/60 }'
```

This gave a total run time of **8.98 hours** (538.68 minutes).  
So the average time for runnigng a full script was **17.95 minutes**.



The routes were then extracted using:

```bash
for i in logs/*.log;
	do cat $i  | grep 'Route' | tail -1 | sed -e 's/Route:        //g' >> routes_temp.txt;
done

sed '$!s/$/,/' routes_temp.txt > routes.txt

rm routes_temp.txt
```





The resulting routes.txt file was then used to generate the graphs.
