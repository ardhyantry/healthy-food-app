@extends('layouts.adminSB')
@section('title', 'Quickbite - Edit Category')

@section('content')
<div class="container">
    <h2>Edit Kategori</h2>
    
    @if ($errors->any())
        <div class="alert alert-danger">
            <ul>
                @foreach ($errors->all() as $error)
                    <li>{{ $error }}</li>
                @endforeach
            </ul>
        </div>
    @endif

    <form action="{{ route('admin.categories.update', $category->id) }}" method="POST">
        @csrf
        @method('PUT')

        <div class="mb-3">
            <label for="name" class="form-label">Nama Kategori</label>
            <input type="text" class="form-control" id="name" name="name" value="{{ old('name', $category->name) }}" required>
        </div>

        <div class="mb-3">
            <label for="image_path" class="form-label">Path Gambar</label>
            <input type="text" class="form-control" id="image_path" name="image_path" value="{{ old('image_path', $category->image_path) }}">
            <br>
            <img src="{{ asset('storage/' . $category->image_path) }}" alt="Current Image" width="150">
        </div>

        <button type="submit" class="btn btn-primary">Simpan Perubahan</button>
        <a href="{{ route('admin.categories.index') }}" class="btn btn-secondary">Kembali</a>
    </form>
</div>
@endsection
