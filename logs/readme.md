# Log files

Genereated with script:

```bash
#!/bin/bash
for i in {1..30};
	do echo $i
	time python3 -u test_script.py > logs/test_$i.log;
done
```

Thr routes were then extracted using:

```bash
#!/bin/bash

for i in logs/*.log;
	do cat $i  | grep 'Route' | tail -1 | sed -e 's/Route:        //g' >> routes_temp.txt;
done

sed '$!s/$/,/' routes_temp.txt > routes.txt

rm routes_temp.txt
```
The resulting routes.txt file was then used to generate the graphs.
