ARG MAYBE_GPU
FROM tensorflow/tensorflow:1.15.2${MAYBE_GPU}-py3-jupyter

# install abcMIDI.
ADD http://abc.sourceforge.net/abcMIDI/original/abcMIDI.tar.gz /
RUN tar -zxf /abcMIDI.tar.gz -C / && cd /abcmidi && make && chmod +x *.exe

# install dependencies.
RUN python3 -m pip install --no-cache-dir --upgrade \
    pip==19.3.1 \
    setuptools==49.6.0

RUN python3 -m pip install --no-cache-dir --upgrade \
    keras==2.2.5 \
    pandas==1.1.1 \
    plotly==4.9.0 \
    pretty-midi==0.2.8 \
    pypianoroll==0.5.3 \
    requests==2.24.0

# install packages to play MIDI in notebooks.
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update -q && apt install -qq --no-install-recommends -y fluidsynth
RUN python3 -m pip install --no-cache-dir --upgrade \
    pyFluidSynth==1.2.5

RUN mkdir /src
ENV PYTHONPATH /src
CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter notebook --notebook-dir=/src --ip 0.0.0.0 --no-browser --allow-root"]
