s.options.numBuffers = 16000;
s.options.memSize = 655360;
s.boot;
s.freqscope;
s.plotTree;
s.scope;

SynthDef(\hydro1, {
	var n = (2..10);
	Out.ar(0,
		(n.collect {arg i; SinOsc.ar( (1 - (1/(i*i))) * 440 )}).sum
	)
}).add;
//Synth(\hydro1);

SynthDef(\hydro3, {
	|out=0,amp=1.0,freq=440|
	var nsize,n = (2..10);
	nsize = n.size;
	Out.ar(0,
		amp * 
		(
			n.collect {arg i; 
				SinOsc.ar( (1.0 - (1/(i*i))) * freq ) +
				SinOsc.ar( ((1/4) - (1/((i+1)*(i+1)))) * freq)
			}).sum / (2 * nsize)
	)
}).add;
//~hydro3 = Synth(\hydro3)
SynthDef(\hydro4, {
	|out=0,amp=1.0,freq=440|
	var nsize,n = (2..10);
	nsize = n.size;
	Out.ar(0,
		amp * 
		(
			n.collect {arg i; 
				SinOsc.ar( (1.0 - (1/(i*i))) * 2*freq ) +
				SinOsc.ar( (1.0 - (1/(i*i))) * freq ) +
				SinOsc.ar( ((1/4) - (1/((i+1)*(i+1)))) * freq)
			}).sum / (3 * nsize)
	)
}).add;

SynthDef(\hydro2, {
	|out=0,amp=1.0,freq=440.0|
	var nsize,n = (2..10);
	nsize = n.size;
	Out.ar(0,
		amp * 
		(
			n.collect {arg i; 
				SinOsc.ar( (1.0 - (1.0/(i*i))) * freq )
			}).sum / nsize
	)
}).add;

~n = 50;
~hydro = ~n.collect { Synth.new([\hydro2,\hydro3,\hydro4].choose) };
~hydro.do { |x| x.set(\amp,1.0/~n) };
~lastx = ~n.collect { 0 };
~lasty = ~n.collect { 0 };
~freqs = ~n.collect { 440 };

~angle = { |x1,y1,x2,y2| 
	((x1*x2) + (y1*y2))  / (sqrt((x1.squared) + (y1.squared)) * sqrt((x2.squared)+(y2.squared)))
};

~angle.(0,0,0,0);
20.collect {|x|~angle.(1.0,0.0,x-10,1.0);};

~listener = {
	|msg|
	var freq = 0.0,nx,ny,lx,ly,angle,scale,dist,index,base;
	msg.postln;
	nx = msg[2]/msg[4];
	ny = msg[3]/msg[5];
	index  = msg[1] % ~n;
	lx = ~lastx[index];
	ly = ~lasty[index];	
	~lastx[index] = nx;
	~lasty[index] = ny;	
	angle = atan2(nx-0.5,ny-0.5) / 3.14;
	//angle = atan2(nx-lx,ny-ly) / 3.14;
	angle.postln;
	//angle = ~angle.(lx+0.5,ly,nx - lx, ny - ly);
	dist = (((nx-lx).squared + (ny-ly).squared).sqrt);
	base = 1000/msg[6];
	scale = base;
	//freq = ((angle + 1.0)/2.0) * scale + base;
	freq =  base + (angle * base / 10);
	~freqs[index] = freq;
	//dist = ((((nx - 0.5)**2) + ((ny - 0.5)**2)).sqrt);
	//freq = scale*dist + 20;
	~hydro[index].set(\freq, freq);
	//~hydro[index].set(\amp, dist/3.0);
	~hydro[index].set(\amp,(1.0/~n));//+(log(1.0+dist)/5.0));
	//~hydro[index].set(\amp, (1.0/~n) * (((nx-lx).squared + (ny-ly).squared).sqrt)  + (1.0/(3*~n)) );
};

OSCFunc.newMatching(~listener, '/flow');


