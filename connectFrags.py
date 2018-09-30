import functionMisc
import numpy as np
def addFragment(fragments,links,nextFrag):
	indexToTest = 0
	while(indexToTest < len(links)):
		removedLink = links[indexToTest]
		remShape = []
		for x in range(indexToTest+1,len(links)):
			remShape = remShape+fragments[x]+links[x]
				
		for x in range(0,indexToTest):
			remShape = remShape + fragments[x] + links[x]	
		remShape = remShape + fragments[indexToTest]	

		works = (not functionMisc.ecross(remShape,nextFrag))
		if not works:
			indexToTest = indexToTest+1
			continue
		a = remShape[0]
		b = remShape[len(remShape)-1]
		c = nextFrag[0]
		d = nextFrag[len(nextFrag)-1]
		

		extra1 = functionMisc.fill([c,b])
		extra1 = extra1[1:len(extra1)-1:1]
		extra2 = functionMisc.fill([a,d])
		extra2 = extra2[1:len(extra2)-1:1]
		fullShape = remShape + extra1[::-1] + nextFrag + extra2[::-1]
		FirstFail =  not ( (not functionMisc.ecross(remShape,extra1))and (not functionMisc.ecross(extra1,extra2))and (not functionMisc.ecross(remShape,extra2))and  (not functionMisc.ecross(nextFrag,extra1))and (not functionMisc.ecross(nextFrag,extra2))  )
		if(not FirstFail):
			newFragments = fragments[:indexToTest+1:1]+[nextFrag]+fragments[indexToTest+1::1]
			newLinks = links[:indexToTest:1] + [extra1[::-1]] + [extra2[::-1]] + links[indexToTest+1::1]
			return[True, newFragments,newLinks,fullShape]
		

		extra1 = functionMisc.fill([c,a])
		extra1 = extra1[1:len(extra1)-1:1]
		extra2 = functionMisc.fill([b,d])
		extra2 = extra2[1:len(extra2)-1:1]
		fullShape =  remShape + extra2 + nextFrag[::-1] + extra1
		SecondFail =  not ( (not functionMisc.ecross(remShape,extra1))and (not functionMisc.ecross(extra1,extra2))and (not functionMisc.ecross(remShape,extra2))and  (not functionMisc.ecross(nextFrag,extra1))and (not functionMisc.ecross(nextFrag,extra2)) )
		if(not SecondFail):
			newFragments = fragments[:indexToTest+1:1]+ [nextFrag[::-1]] +fragments[indexToTest+1::1]
			newLinks = links[:indexToTest:1]+[extra2]+[extra1]+links[indexToTest+1::1]

			return[True, newFragments,newLinks,fullShape]
		indexToTest = indexToTest+1
	return [False]


