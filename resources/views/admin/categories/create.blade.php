@extends('layouts.adminSB')
@section('title', 'Quickbite - Create Category')

@section('content')
<div class="container">
    <h2>Tambah Kategori Baru</h2>

    @if ($errors->any())
        <div class="alert alert-danger">
            <ul>
                @foreach ($errors->all() as $error)
                    <li>{{ $error }}</li>
                @endforeach
            </ul>
        </div>
    @endif

    <form action="{{ route('admin.categories.store') }}" method="POST">
        @csrf

        <div class="mb-3">
            <label for="name" class="form-label">Nama Kategori</label>
            <input type="text" class="form-control" id="name" name="name" value="{{ old('name') }}" required>
        </div>

        <div class="mb-3">
            <label for="image_path" class="form-label">Path Gambar</label>
            <input type="text" class="form-control" id="image_path" name="image_path" value="{{ old('image_path') }}">
            <small class="text-muted">Contoh: <code>categories/dessert.jpg</code></small>
        </div>

        <button type="submit" class="btn btn-success">Simpan</button>
        <a href="{{ route('admin.categories.index') }}" class="btn btn-secondary">Kembali</a>
    </form>
</div>
@endsection
