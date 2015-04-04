s.boot;

~listener = {
	|msg|
	msg.postln;
	(freq: 20 + 2000*(msg[2] / msg[4])).play();
};
(freq: 100).play()

OSCFunc.newMatching(~listener, '/flow');
OSCFunc.newMatching(~listener, '/flow/');

