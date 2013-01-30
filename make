CPP_FILES := $(wildcard src/*.cpp)
OBJFILES := $(patsubst src/%.cpp, obj/%.o, $CPP_FILES)

