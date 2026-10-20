package scene_test

import (
	"errors"
	"testing"

	domain "github.com/MarkJulian19/path_trace_golang/internal/entities/scene"
	scenecase "github.com/MarkJulian19/path_trace_golang/internal/usecases/scene"
)

var (
	errTestRepository     = errors.New("repository failure")
	errExpectedRepository = errors.New("expected repository error")
	errNotWrapped         = errors.New("repository error was not wrapped")
)

type testRepository struct {
	value   *domain.Scene
	loadErr error
	saveErr error
	loaded  string
	saved   string
}

func (r *testRepository) Load(path string) (*domain.Scene, error) {
	r.loaded = path
	if r.loadErr != nil {
		return nil, r.loadErr
	}
	return r.value, nil
}

func (r *testRepository) Save(path string, value *domain.Scene) error {
	r.saved = path
	r.value = value
	return r.saveErr
}

func TestSceneUsecaseLoadAndSave(t *testing.T) {
	t.Parallel()
	value := &domain.Scene{Name: "scene"}
	repository := &testRepository{value: value}
	usecase := scenecase.New(repository)

	loaded, err := usecase.Load("input.json")
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	if loaded != value {
		t.Fatal("Load() returned a different scene")
	}
	if repository.loaded != "input.json" {
		t.Fatalf("loaded path = %q, want input.json", repository.loaded)
	}

	toSave := &domain.Scene{Name: "saved"}
	err = usecase.Save("output.json", toSave)
	if err != nil {
		t.Fatalf("Save() error = %v", err)
	}
	if repository.saved != "output.json" {
		t.Fatalf("saved path = %q, want output.json", repository.saved)
	}
	if repository.value != toSave {
		t.Fatal("Save() passed a different scene")
	}
}

func TestSceneUsecaseWrapsRepositoryErrors(t *testing.T) {
	t.Parallel()
	loadUsecase := scenecase.New(&testRepository{loadErr: errTestRepository})
	err := assertWrappedError(func() error {
		_, loadErr := loadUsecase.Load("broken.json")
		return loadErr
	})
	if err != nil {
		t.Fatal(err)
	}

	saveUsecase := scenecase.New(&testRepository{saveErr: errTestRepository})
	err = assertWrappedError(func() error {
		return saveUsecase.Save("broken.json", &domain.Scene{})
	})
	if err != nil {
		t.Fatal(err)
	}
}

func assertWrappedError(operation func() error) error {
	err := operation()
	if err == nil {
		return errExpectedRepository
	}
	if !errors.Is(err, errTestRepository) {
		return errNotWrapped
	}
	return nil
}

func BenchmarkSceneUsecaseLoad(b *testing.B) {
	repository := &testRepository{value: &domain.Scene{Name: "benchmark"}}
	usecase := scenecase.New(repository)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		benchmarkSceneValue, errBenchmarkScene = usecase.Load("scene.json")
	}
}

func BenchmarkSceneUsecaseNew(b *testing.B) {
	repository := &testRepository{}
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		benchmarkSceneUsecase = scenecase.New(repository)
	}
}

func BenchmarkSceneUsecaseSave(b *testing.B) {
	repository := &testRepository{}
	usecase := scenecase.New(repository)
	value := &domain.Scene{Name: "benchmark"}
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		errBenchmarkScene = usecase.Save("scene.json", value)
	}
}

var (
	benchmarkSceneValue   *domain.Scene
	benchmarkSceneUsecase *scenecase.Scene
	errBenchmarkScene     error
)
