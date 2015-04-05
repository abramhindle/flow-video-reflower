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

~hydro = [Synth.new(\hydro3),Synth.new(\hydro2),Synth.new(\hydro4) ];  //3.collect { Synth.new(\hydro2) };
~hydro.do { |x| x.set(\amp,0.1) };
~listener = {
	|msg|
	var freq = 0.0,nx,ny,scale,dist;
	msg.postln;
	nx = msg[2]/msg[4];
	ny = msg[3]/msg[5];
	scale = 2000.0/msg[6];
	dist = ((((nx - 0.5)**2) + ((ny - 0.5)**2)).sqrt);
	freq = scale*dist + 20;
	~hydro[msg[1] % 3].set(\freq, freq);
	~hydro[msg[1] % 3].set(\amp, dist/3.0);
};

OSCFunc.newMatching(~listener, '/flow');
OSCFunc.newMatching(~listener, '/flow/');

